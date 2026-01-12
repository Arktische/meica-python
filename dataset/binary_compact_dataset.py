import os
import torch
import io
import logging
from torch.utils.data import Dataset
import numpy as np

logger = logging.getLogger(__name__)


class BinaryCompactDataset(Dataset):
    def __init__(
        self,
        cache_dir: str,
    ):
        """
        Args:
            cache_dir: Directory where the preprocessed binary files and metadata.pth are stored.
        """
        self.cache_dir = cache_dir
        self._file_handles = {}
        self._pid = None
        
        metadata_path = os.path.join(cache_dir, "metadata.pth")
        if not os.path.exists(metadata_path):
            raise RuntimeError(
                f"Metadata file 'metadata.pth' not found in {cache_dir}. "
                "Please run the Preprocessor first to generate the cache and metadata."
            )
        
        logger.info(f"Loading dataset items from {metadata_path}")
        self.item_names = torch.load(metadata_path, weights_only=True)

        self.offsets = {}
        self.length = 0

        if not self._check_cache_exists():
            raise RuntimeError(
                f"Preprocessed cache not found in {cache_dir}. "
                "Please run the Preprocessor first."
            )

        self._load_indices()

    def _check_cache_exists(self) -> bool:
        for name in self.item_names:
            data_path = os.path.join(self.cache_dir, f"{name}.bin")
            index_path = os.path.join(self.cache_dir, f"{name}.idx")
            if not os.path.exists(data_path) or not os.path.exists(index_path):
                return False
        return True

    def _load_indices(self):
        for name in self.item_names:
            # We load the offsets into memory (numpy array)
            # Index file contains N+1 64-bit integers
            index_path = os.path.join(self.cache_dir, f"{name}.idx")

            with open(index_path, "rb") as f:
                data = f.read()
                # Load as numpy array efficiently
                # 'Q' is unsigned long long (8 bytes).
                offsets_np = np.frombuffer(data, dtype=np.uint64)
                self.offsets[name] = offsets_np

        # Validate lengths
        lengths = [len(off) - 1 for off in self.offsets.values()]
        if not all(l == lengths[0] for l in lengths):
            raise RuntimeError(f"Mismatch in dataset item lengths: {lengths}")

        self.length = lengths[0]

    def __len__(self):
        return self.length

    def _get_file_handle(self, item_name: str):
        """Get or open a file handle for the given item, ensuring process safety."""
        current_pid = os.getpid()
        if self._pid != current_pid:
            # Process has changed (e.g., forked), close old handles and reset
            for f in self._file_handles.values():
                try:
                    f.close()
                except Exception:
                    pass
            self._file_handles = {}
            self._pid = current_pid

        if item_name not in self._file_handles:
            data_path = os.path.join(self.cache_dir, f"{item_name}.bin")
            self._file_handles[item_name] = open(data_path, "rb")
        
        return self._file_handles[item_name]

    def close(self):
        """Close all open file handles."""
        for f in self._file_handles.values():
            try:
                f.close()
            except Exception:
                pass
        self._file_handles = {}

    def __del__(self):
        self.close()

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise IndexError(f"Index {idx} out of range")

        result = {}
        for name in self.item_names:
            offsets = self.offsets[name]
            start = int(offsets[idx])
            end = int(offsets[idx + 1])
            length = end - start

            # Use cached file handle
            f = self._get_file_handle(name)
            f.seek(start)
            data = f.read(length)

            # Deserialize
            buffer = io.BytesIO(data)
            tensor = torch.load(buffer, weights_only=True)
            result[name] = tensor

        return result
