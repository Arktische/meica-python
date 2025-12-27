from torch.utils.data import Dataset
import torch
import os
from pathlib import Path
from PIL import Image
from typing import Callable, List, Optional, Dict, Union, Tuple, Any
import numpy as np
from tqdm import tqdm
import glob
import multiprocessing
import json
import struct
import pickle
import io
import zlib
import shutil
import torch.distributed as dist

def get_dist_info():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1

class DatasetFileMatcher:
    def __init__(self, 
                 img_dir: str, 
                 control_img_dir: str, 
                 instruction_dir: str, 
                 img_exts: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp', '.webp'),
                 txt_exts: Tuple[str, ...] = ('.txt', '.json')):
        self.img_dir = Path(img_dir)
        self.control_img_dir = Path(control_img_dir)
        self.instruction_dir = Path(instruction_dir)
        self.img_exts = img_exts
        self.txt_exts = txt_exts
        
    def scan_and_match(self) -> List[Dict[str, str]]:
        # Find all files
        img_files = self._get_files(self.img_dir, self.img_exts)
        control_files = self._get_files(self.control_img_dir, self.img_exts)
        instruction_files = self._get_files(self.instruction_dir, self.txt_exts)
        
        # Match by stem (base name without extension)
        common_stems = set(img_files.keys()) & set(control_files.keys()) & set(instruction_files.keys())
        
        matched_samples = []
        for stem in sorted(list(common_stems)):
            matched_samples.append({
                'img_path': str(img_files[stem]),
                'control_path': str(control_files[stem]),
                'instruction_path': str(instruction_files[stem]),
                'stem': stem
            })
            
        print(f"Found {len(matched_samples)} matched samples out of {len(img_files)} images.")
        return matched_samples

    def _get_files(self, directory: Path, extensions: Tuple[str, ...]) -> Dict[str, Path]:
        if not directory.exists():
            return {}
        files = {}
        for ext in extensions:
            for p in directory.glob(f"*{ext}"):
                files[p.stem] = p
        return files

class RawDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        sample = self.samples[i]
        
        try:
            img = Image.open(sample['img_path']).convert('RGB')
        except Exception as e:
            print(f"Error loading image {sample['img_path']}: {e}")
            img = Image.new('RGB', (256, 256))

        try:
            control_img = Image.open(sample['control_path']).convert('RGB')
        except Exception as e:
            print(f"Error loading control image {sample['control_path']}: {e}")
            control_img = Image.new('RGB', (256, 256))

        text = ""
        try:
            path = sample['instruction_path']
            with open(path, 'r', encoding='utf-8') as f:
                if path.endswith('.json'):
                    data = json.load(f)
                    text = json.dumps(data)
                else:
                    text = f.read().strip()
        except Exception as e:
            print(f"Error loading text {path}: {e}")
            
        return {'image': img, 'control_image': control_img, 'text': text, 'stem': sample['stem']}

def identity_collate(batch):
    return batch

class BinaryIndex:
    # Magic: MEICAIDX (8 bytes), Version: 1 (1 byte) -> Total 9 bytes header
    MAGIC = b'MEICAIDX'
    VERSION = 1
    HEADER_FMT = '<8sB Q' # Magic, Version, Count
    HEADER_SIZE = struct.calcsize(HEADER_FMT)
    # Record: chunk_id (H), offset (Q), length (I), checksum (I)
    RECORD_FMT = '<H Q I I' 
    RECORD_SIZE = struct.calcsize(RECORD_FMT)

    @staticmethod
    def save(path: str, records: List[Dict[str, Any]]):
        """
        Records: list of dict with 'chunk_id', 'offset', 'length', 'checksum'
        """
        count = len(records)
        with open(path, 'wb') as f:
            # Write Header
            f.write(struct.pack(BinaryIndex.HEADER_FMT, BinaryIndex.MAGIC, BinaryIndex.VERSION, count))
            
            # Write Records
            for r in records:
                f.write(struct.pack(
                    BinaryIndex.RECORD_FMT, 
                    r['chunk_id'], 
                    r['offset'], 
                    r['length'], 
                    r['checksum']
                ))

    @staticmethod
    def load(path: str) -> List[Dict[str, Any]]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Index file not found: {path}")
            
        with open(path, 'rb') as f:
            # Read Header
            header_data = f.read(BinaryIndex.HEADER_SIZE)
            if len(header_data) < BinaryIndex.HEADER_SIZE:
                raise ValueError("Index file too short")
                
            magic, version, count = struct.unpack(BinaryIndex.HEADER_FMT, header_data)
            
            if magic != BinaryIndex.MAGIC:
                raise ValueError(f"Invalid magic: {magic}")
            if version != BinaryIndex.VERSION:
                raise ValueError(f"Unsupported version: {version}")
                
            # Read Records
            # We can read all at once for speed using numpy or just loop
            # For simplicity and safety (endianness), loop with struct
            records = []
            
            # Estimate file size check
            expected_size = BinaryIndex.HEADER_SIZE + count * BinaryIndex.RECORD_SIZE
            f.seek(0, 2)
            actual_size = f.tell()
            if actual_size != expected_size:
                print(f"Warning: Index file size mismatch. Expected {expected_size}, got {actual_size}")
                
            f.seek(BinaryIndex.HEADER_SIZE)
            
            # Read all records bytes
            records_data = f.read(count * BinaryIndex.RECORD_SIZE)
            
            # Unpack
            # It's faster to iter_unpack if python version supports it (3.4+)
            try:
                iter_recs = struct.iter_unpack(BinaryIndex.RECORD_FMT, records_data)
                for chunk_id, offset, length, checksum in iter_recs:
                    records.append({
                        'chunk_id': chunk_id,
                        'offset': offset,
                        'length': length,
                        'checksum': checksum
                    })
            except AttributeError:
                 # Fallback for older python if needed (unlikely here)
                 for i in range(count):
                     chunk = records_data[i*BinaryIndex.RECORD_SIZE : (i+1)*BinaryIndex.RECORD_SIZE]
                     chunk_id, offset, length, checksum = struct.unpack(BinaryIndex.RECORD_FMT, chunk)
                     records.append({
                        'chunk_id': chunk_id,
                        'offset': offset,
                        'length': length,
                        'checksum': checksum
                    })
            
            return records

class ImageToImageDataset(Dataset):
    def __init__(self, 
                 img_dir: str, 
                 control_img_dir: str, 
                 instruction_dir: str, 
                 img_exts: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.webp'),
                 txt_exts: Tuple[str, ...] = ('.txt',),
                 precompute_embeddings: bool = False,
                 encode_latents: Optional[Callable[[Image.Image], torch.Tensor]] = None, 
                 encode_prompt: Optional[Callable[[str], torch.Tensor]] = None,
                 cache_dir: Optional[str] = None,
                 std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
                 mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
                 binary_path: Optional[str] = None,
                 binary_chunk_size: int = 512 * 1024 * 1024):
        
        self.std = torch.tensor(std).view(3, 1, 1)
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.encode_latents = encode_latents
        self.encode_prompt = encode_prompt
        self.precompute_embeddings = precompute_embeddings
        self.cache_dir = cache_dir
        self.binary_path = binary_path
        self.binary_chunk_size = binary_chunk_size
        
        # Determine mode
        self.mode = 'source' # Default
        
        if binary_path:
            # Check if binary dataset exists and is valid
            index_path = os.path.join(binary_path, "index.bin")
            if os.path.exists(index_path):
                try:
                    self.index = BinaryIndex.load(index_path)
                    print(f"Loaded binary index with {len(self.index)} samples.")
                    self.mode = 'binary'
                except Exception as e:
                    print(f"Failed to load binary index: {e}. Re-creating...")
                    self.mode = 'source'
            else:
                print(f"Binary path {binary_path} provided but index not found. Will create from source.")
                self.mode = 'source'
                
        if self.mode == 'source':
            self.matcher = DatasetFileMatcher(img_dir, control_img_dir, instruction_dir, img_exts, txt_exts)
            self.samples = self.matcher.scan_and_match()
            
            if precompute_embeddings:
                if cache_dir is None:
                    raise ValueError("cache_dir must be provided when precompute_embeddings is True")
                self._precompute_and_cache()
            
            # If binary_path was requested but we are in source mode, it means we need to export
            if binary_path:
                print("Exporting dataset to binary format...")
                self.export_to_binary(binary_path, chunk_size=self.binary_chunk_size)
                # Switch to binary mode
                self.index = BinaryIndex.load(os.path.join(binary_path, "index.bin"))
                self.mode = 'binary'
                # Clear source samples to save memory if desired
                # self.samples = [] 


    def __len__(self):
        if self.mode == 'binary':
            return len(self.index)
        return len(self.samples)

    def _get_binary_item(self, idx):
        record = self.index[idx]
        chunk_file = os.path.join(self.binary_path, f"chunk_{record['chunk_id']}.bin")
        offset = record['offset']
        length = record['length']
        checksum = record['checksum']
        
        with open(chunk_file, 'rb') as f:
            f.seek(offset)
            data = f.read(length)
            
        # Verify checksum
        if zlib.crc32(data) != checksum:
            raise ValueError(f"Data corruption detected for sample {idx} in {chunk_file}")
            
        sample = pickle.loads(data)
        return sample

    def export_to_binary(self, output_dir: str, chunk_size: int = 512 * 1024 * 1024):
        """
        Export the current dataset (including precomputed embeddings if enabled) to binary format.
        """
        rank, world_size = get_dist_info()

        if rank == 0:
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            os.makedirs(output_dir, exist_ok=True)
        
        if world_size > 1:
            dist.barrier()
        
        # Distribute indices
        all_indices = list(range(len(self)))
        # Distribute work round-robin style
        my_indices = [idx for i, idx in enumerate(all_indices) if i % world_size == rank]
        
        current_chunk_idx = 0
        chunk_filename = f"chunk_{rank}_{current_chunk_idx}.bin"
        f = open(os.path.join(output_dir, chunk_filename), 'wb')
        current_file_size = 0
        
        index_records = []
        
        for idx in tqdm(my_indices, desc=f"Exporting Rank {rank}"):
            sample_data = self[idx]
            serialized = pickle.dumps(sample_data)
            size = len(serialized)
            crc = zlib.crc32(serialized)
            
            # Check if we need to rotate chunk
            if current_file_size + size > chunk_size and current_file_size > 0:
                f.close()
                current_chunk_idx += 1
                chunk_filename = f"chunk_{rank}_{current_chunk_idx}.bin"
                f = open(os.path.join(output_dir, chunk_filename), 'wb')
                current_file_size = 0
            
            # Write data
            f.write(serialized)
            
            # Update index
            index_records.append({
                'chunk_id': current_chunk_idx,
                'offset': current_file_size,
                'length': size,
                'checksum': crc,
                'original_idx': idx
            })
            
            current_file_size += size
            
        f.close()
        
        # Save partial index
        with open(os.path.join(output_dir, f"index_part_{rank}.pkl"), 'wb') as f:
            pickle.dump(index_records, f)

        if world_size > 1:
            dist.barrier()
            
        # Aggregation (Rank 0 only)
        if rank == 0:
            print("Aggregating chunks and indices...")
            final_records = []
            global_chunk_id = 0
            chunk_map = {} # (rank, local_chunk_id) -> global_chunk_id
            
            for r in range(world_size):
                part_path = os.path.join(output_dir, f"index_part_{r}.pkl")
                if not os.path.exists(part_path):
                    continue
                    
                with open(part_path, 'rb') as f:
                    part_records = pickle.load(f)
                
                for record in part_records:
                    local_cid = record['chunk_id']
                    key = (r, local_cid)
                    
                    if key not in chunk_map:
                        chunk_map[key] = global_chunk_id
                        
                        # Rename file
                        src = os.path.join(output_dir, f"chunk_{r}_{local_cid}.bin")
                        dst = os.path.join(output_dir, f"chunk_{global_chunk_id}.bin")
                        if os.path.exists(src):
                            os.rename(src, dst)
                        
                        global_chunk_id += 1
                    
                    record['chunk_id'] = chunk_map[key]
                    final_records.append(record)
                
                os.remove(part_path)
            
            # Sort by original index to preserve dataset order
            final_records.sort(key=lambda x: x['original_idx'])
            
            # Save binary index
            BinaryIndex.save(os.path.join(output_dir, "index.bin"), final_records)
            
            print(f"Exported {len(final_records)} samples to {output_dir} in {global_chunk_id} chunks.")

    def _load_image(self, path: str) -> torch.Tensor:
        try:
            img = Image.open(path).convert('RGB')
            # Convert to tensor and normalize
            img_t = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0
            img_t = (img_t - self.mean) / self.std
            return img_t
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            return torch.zeros(3, 256, 256) # Return dummy or raise

    def _load_text(self, path: str) -> str:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                if path.endswith('.json'):
                    data = json.load(f)
                    # Assume simple json structure or specific key? 
                    # For now return dump or specific field if known.
                    # Requirement says "support common... text formats".
                    return json.dumps(data) 
                return f.read().strip()
        except Exception as e:
            print(f"Error loading text {path}: {e}")
            return ""

    def __getitem__(self, idx):
        if self.mode == 'binary':
            return self._get_binary_item(idx)

        sample = self.samples[idx]
        stem = sample['stem']
        
        # Try to load from cache if enabled
        if self.precompute_embeddings and self.cache_dir:
            latent_path = os.path.join(self.cache_dir, f"{stem}_latent.pt")
            control_latent_path = os.path.join(self.cache_dir, f"{stem}_control_latent.pt")
            embed_path = os.path.join(self.cache_dir, f"{stem}_embed.pt")
            
            # We require all cache files to be present
            if os.path.exists(latent_path) and os.path.exists(embed_path) and os.path.exists(control_latent_path):
                try:
                    latent = torch.load(latent_path)
                    control_latent = torch.load(control_latent_path)
                    embed = torch.load(embed_path)
                    return {
                        'latent': latent, 
                        'control_latent': control_latent, 
                        'embedding': embed, 
                        'stem': stem
                    }
                except Exception as e:
                    print(f"Failed to load cache for {stem}: {e}")

        # Live loading
        img_path = sample['img_path']
        control_path = sample['control_path']
        instruction_path = sample['instruction_path']
        
        img_tensor = self._load_image(img_path)
        control_tensor = self._load_image(control_path)
        instruction_text = self._load_text(instruction_path)
        
        result = {
            'image': img_tensor,
            'control_image': control_tensor,
            'instruction': instruction_text,
            'stem': stem
        }

        if self.encode_latents:
            # Load PIL for encoding
            img_pil = Image.open(img_path).convert('RGB')
            control_pil = Image.open(control_path).convert('RGB')
            
            with torch.no_grad():
                result['latent'] = self.encode_latents(img_pil)
                result['control_latent'] = self.encode_latents(control_pil)

        if self.encode_prompt:
            with torch.no_grad():
                result['embedding'] = self.encode_prompt(instruction_text)

        return result

    def _precompute_and_cache(self):
        rank, world_size = get_dist_info()

        if rank == 0:
            os.makedirs(self.cache_dir, exist_ok=True)
        
        if world_size > 1:
            dist.barrier()
        
        # Distribute work first to avoid scanning all files on all ranks
        all_indices = range(len(self.samples))
        my_indices = [i for i in all_indices if i % world_size == rank]
        
        # Identify missing cache for my indices only
        my_missing_indices = []
        for idx in my_indices:
            sample = self.samples[idx]
            stem = sample['stem']
            latent_path = os.path.join(self.cache_dir, f"{stem}_latent.pt")
            control_latent_path = os.path.join(self.cache_dir, f"{stem}_control_latent.pt")
            embed_path = os.path.join(self.cache_dir, f"{stem}_embed.pt")
            
            if not (os.path.exists(latent_path) and os.path.exists(control_latent_path) and os.path.exists(embed_path)):
                my_missing_indices.append(idx)
        
        if not my_missing_indices:
            print(f"[Rank {rank}] All assigned samples already cached.")
            if world_size > 1:
                dist.barrier()
            return

        print(f"[Rank {rank}] Pre-computing {len(my_missing_indices)} samples...")
        
        # Filter samples that need processing
        missing_samples = [self.samples[i] for i in my_missing_indices]
        
        # Use a DataLoader to fetch data in parallel
        # We use the module-level RawDataset to avoid pickling issues
        raw_ds = RawDataset(missing_samples)
        
        loader = torch.utils.data.DataLoader(
            raw_ds, 
            batch_size=1, # Process one by one for encoding safety or batch if encoders support it
            num_workers=min(os.cpu_count(), 8), 
            collate_fn=identity_collate
        )

        for batch in tqdm(loader, desc=f"Caching Rank {rank}"):
            # Batch size is 1, so extract item
            item = batch[0]
            stem = item['stem']
            img = item['image']
            control_img = item['control_image']
            text = item['text']
            
            latent_path = os.path.join(self.cache_dir, f"{stem}_latent.pt")
            control_latent_path = os.path.join(self.cache_dir, f"{stem}_control_latent.pt")
            embed_path = os.path.join(self.cache_dir, f"{stem}_embed.pt")
            
            if self.encode_latents:
                if not os.path.exists(latent_path):
                    try:
                        with torch.no_grad():
                            latent = self.encode_latents(img)
                        torch.save(latent, latent_path)
                    except Exception as e:
                        print(f"Error encoding latent for {stem}: {e}")
                
                if not os.path.exists(control_latent_path):
                    try:
                        with torch.no_grad():
                            control_latent = self.encode_latents(control_img)
                        torch.save(control_latent, control_latent_path)
                    except Exception as e:
                        print(f"Error encoding control latent for {stem}: {e}")

            if self.encode_prompt and not os.path.exists(embed_path):
                try:
                    with torch.no_grad():
                        embed = self.encode_prompt(text)
                    torch.save(embed, embed_path)
                except Exception as e:
                    print(f"Error encoding prompt for {stem}: {e}")
        
        if world_size > 1:
            dist.barrier()

