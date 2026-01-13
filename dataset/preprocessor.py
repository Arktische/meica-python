import os
import torch
import torch.distributed as dist
import struct
import math
import io
import logging
import gc
from typing import Callable, Union, Optional, Tuple, Dict, List, Any
from torch.multiprocessing.spawn import spawn
from tqdm import tqdm

logger = logging.getLogger(__name__)


class DatasetItem:
    def __init__(
        self,
        match_key: Callable[[str], Optional[str]],
        transform: Callable[..., Union[torch.Tensor, Tuple, Dict[str, torch.Tensor]]],
        name: str,
    ):
        """
        Args:
            match_key: Function to extract a unique key from the file path.
                       Returns the key string, or None if the file should be ignored.
            transform: Function to transform the file into a tensor.
                       Args: file_path, device, **kwargs. Returns: Tensor, Tuple of Tensors, or Dict of Tensors.
                       kwargs contains other item paths with the same match_key.
            name: Name of the item (key in the returned dictionary).
        """
        self.match_key = match_key
        self.transform = transform
        self.name = name


class Preprocessor:
    def __init__(
        self,
        root_dirs: List[str],
        items: List[DatasetItem],
        cache_dir: str = os.path.join(os.getcwd(), ".preprocess_cache"),
        force_rebuild: bool = False,
        num_workers: Optional[int] = None,
        **dependencies,
    ):
        self.root_dirs = root_dirs
        self.items = items
        self.cache_dir = cache_dir
        self.force_rebuild = force_rebuild
        self.num_workers = num_workers
        self.dependencies = dependencies

    def _check_cache_exists(self) -> bool:
        metadata_path = os.path.join(self.cache_dir, "metadata.pth")
        if not os.path.exists(metadata_path):
            return False
        for item in self.items:
            data_path = os.path.join(self.cache_dir, f"{item.name}.bin")
            index_path = os.path.join(self.cache_dir, f"{item.name}.idx")
            if not os.path.exists(data_path) or not os.path.exists(index_path):
                return False
        return True

    def _scan_files(self) -> Tuple[List[str], Dict[str, Dict[str, str]]]:
        """Scans directories and returns a sorted list of unique keys that have all required items."""
        logger.info("Scanning files...")
        temp_map: Dict[str, Dict[str, str]] = {}  # key -> {item_name -> path}

        for root_dir in self.root_dirs:
            # Walk through all files
            for root, _, files in os.walk(root_dir):
                for file in files:
                    full_path = os.path.join(root, file)
                    for item in self.items:
                        try:
                            key = item.match_key(full_path)
                            if key:
                                if key not in temp_map:
                                    temp_map[key] = {}
                                temp_map[key][item.name] = full_path
                        except Exception:
                            continue

        # Filter complete samples
        valid_keys = []
        required_names = set(item.name for item in self.items)

        for key, paths in temp_map.items():
            if set(paths.keys()) == required_names:
                valid_keys.append(key)

        valid_keys.sort()
        logger.info(f"Found {len(valid_keys)} valid samples.")
        return valid_keys, temp_map

    @staticmethod
    def _to_cpu(data):
        """Recursively move tensors in a nested structure (dict, list, tuple) to CPU."""
        if isinstance(data, torch.Tensor):
            return data.cpu()
        elif isinstance(data, dict):
            return {k: Preprocessor._to_cpu(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [Preprocessor._to_cpu(v) for v in data]
        elif isinstance(data, tuple):
            return tuple(Preprocessor._to_cpu(v) for v in data)
        return data

    @staticmethod
    def _worker(
        rank: int,
        chunks: List[List[str]],
        file_map: Dict[str, Dict[str, str]],
        items: List[DatasetItem],
        cache_dir: str,
        num_gpus: int,
        dependencies: Optional[Dict[str, Any]] = None,
    ):
        keys = chunks[rank]
        # Assign device: if GPUs available, round robin or 1:1.
        if num_gpus > 0:
            device_id = rank % num_gpus
            device = torch.device(f"cuda:{device_id}")
        else:
            device = torch.device("cpu")

        # Prepare dependencies: move tensors and nn.Modules to device
        prepared_deps = {}
        if dependencies:
            for k, v in dependencies.items():
                if isinstance(v, (torch.Tensor, torch.nn.Module)):
                    prepared_deps[k] = v.to(device)
                else:
                    prepared_deps[k] = v

        # Open handles for each item
        files = {}
        indices = {}
        current_offsets = {}

        try:
            for item in items:
                f_bin = open(os.path.join(cache_dir, f"{item.name}_{rank}.bin"), "wb")
                f_idx = open(os.path.join(cache_dir, f"{item.name}_{rank}.idx"), "wb")
                files[item.name] = f_bin
                indices[item.name] = f_idx
                current_offsets[item.name] = 0

                # Write initial offset (0)
                f_idx.write(struct.pack("Q", 0))

            for key in tqdm(keys, position=rank, desc=f"Worker {rank}"):
                paths = file_map[key]
                for item in items:
                    try:
                        path = paths[item.name]
                        # Transform
                        # Pass other paths and dependencies as kwargs
                        kwargs = {k: v for k, v in paths.items() if k != item.name}
                        kwargs.update(prepared_deps)
                        data = item.transform(path, device, **kwargs)

                        # Move to CPU for serialization to avoid CUDA sharing issues in pickle/storage
                        data = Preprocessor._to_cpu(data)

                        # Serialize
                        # Use torch.save approach or pickle.
                        # Pickle is general. For tensors, io.BytesIO + torch.save is robust.
                        buffer = io.BytesIO()
                        torch.save(data, buffer)
                        serialized_data = buffer.getvalue()
                        length = len(serialized_data)

                        # Write data
                        files[item.name].write(serialized_data)

                        # Update offset
                        current_offsets[item.name] += length

                        # Write NEW offset (end of this item, start of next)
                        indices[item.name].write(
                            struct.pack("Q", current_offsets[item.name])
                        )
                    except Exception as e:
                        logger.error(
                            f"Error processing item {item.name} for key {key}: {e}"
                        )
                        raise e

        finally:
            for f in files.values():
                f.close()
            for f in indices.values():
                f.close()

    def _merge_files(self, num_workers: int):
        # For each item type
        for item in self.items:
            final_bin_path = os.path.join(self.cache_dir, f"{item.name}.bin")
            final_idx_path = os.path.join(self.cache_dir, f"{item.name}.idx")

            with open(final_bin_path, "wb") as f_out_bin, open(
                final_idx_path, "wb"
            ) as f_out_idx:

                total_offset = 0
                # Write the initial 0 offset for the merged index file
                f_out_idx.write(struct.pack("Q", 0))

                for rank in range(num_workers):
                    part_bin_path = os.path.join(
                        self.cache_dir, f"{item.name}_{rank}.bin"
                    )
                    part_idx_path = os.path.join(
                        self.cache_dir, f"{item.name}_{rank}.idx"
                    )

                    # Merge Binary Data
                    with open(part_bin_path, "rb") as f_in_bin:
                        # Chunked copy to avoid memory overflow
                        while True:
                            chunk = f_in_bin.read(1024 * 1024 * 10)  # 10MB chunks
                            if not chunk:
                                break
                            f_out_bin.write(chunk)

                    # Merge Index Data
                    # Read and adjust offsets from part file in chunks
                    with open(part_idx_path, "rb") as f_in_idx:
                        # Skip the first offset (always 0) of the part file,
                        # because it's already covered by the previous rank's last offset
                        # or the initial 0.
                        f_in_idx.seek(8)

                        while True:
                            chunk = f_in_idx.read(
                                1024 * 1024
                            )  # 1MB of offsets (128k samples)
                            if not chunk:
                                break

                            count = len(chunk) // 8
                            part_offsets = struct.unpack(f"{count}Q", chunk)

                            for offset in part_offsets:
                                adjusted_offset = offset + total_offset
                                f_out_idx.write(struct.pack("Q", adjusted_offset))

                        # The last offset of this part becomes the new total_offset base
                        # We need to get the absolute last offset from this part file.
                        f_in_idx.seek(0, os.SEEK_END)
                        f_in_idx.seek(f_in_idx.tell() - 8)
                        last_offset_data = f_in_idx.read(8)
                        last_offset = struct.unpack("Q", last_offset_data)[0]
                        total_offset += last_offset

                    # Clean up
                    os.remove(part_bin_path)
                    os.remove(part_idx_path)

    def run(self):
        # Distributed check: only run on rank 0
        rank = 0
        is_distributed = dist.is_available() and dist.is_initialized()
        if is_distributed:
            rank = dist.get_rank()

        if rank != 0:
            logger.info(f"Rank {rank} skipping build, waiting for rank 0.")
            dist.barrier()
            return

        if not self.force_rebuild and self._check_cache_exists():
            logger.info("Cache exists, skipping build.")
            if is_distributed:
                dist.barrier()
            return

        os.makedirs(self.cache_dir, exist_ok=True)

        logger.info("Building dataset...")
        valid_keys, file_map = self._scan_files()

        if not valid_keys:
            if is_distributed:
                dist.barrier()
            raise RuntimeError("No valid samples found matching all DatasetItems.")

        num_gpus = torch.cuda.device_count()
        if self.num_workers is not None:
            num_workers = self.num_workers
        else:
            num_workers = max(1, num_gpus)

        # Split keys into chunks
        chunk_size = math.ceil(len(valid_keys) / num_workers)
        chunks = [
            valid_keys[i : i + chunk_size]
            for i in range(0, len(valid_keys), chunk_size)
        ]

        # Ensure we don't have empty chunks if valid_keys < num_workers
        chunks = [c for c in chunks if c]
        num_workers = len(chunks)

        logger.info(
            f"Starting distributed preprocessing with {num_workers} workers on {num_gpus} GPUs..."
        )

        # Use spawn for CUDA compatibility
        spawn(
            self._worker,
            args=(
                chunks,
                file_map,
                self.items,
                self.cache_dir,
                num_gpus,
                self.dependencies,
            ),
            nprocs=num_workers,
            join=True,
        )

        logger.info("Merging distributed files...")
        self._merge_files(num_workers)

        # Save items metadata for BinaryCompactDataset to load without explicit items
        metadata_path = os.path.join(self.cache_dir, "metadata.pth")
        item_names = [item.name for item in self.items]
        torch.save(item_names, metadata_path)
        logger.info(f"Metadata saved to {metadata_path}")

        # Clear dependencies to release memory/VRAM
        self.dependencies.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Dependencies cleared and memory released.")

        logger.info("Dataset build complete.")
        if is_distributed:
            dist.barrier()
