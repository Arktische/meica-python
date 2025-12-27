import os
import torch
import torch.distributed as dist
import shutil
import tempfile
import unittest
from pathlib import Path
from PIL import Image
from i2i_dataset import ImageToImageDataset, BinaryIndex

def setup_dummy_data(root_dir, num_samples=10):
    img_dir = os.path.join(root_dir, 'images')
    control_dir = os.path.join(root_dir, 'controls')
    instruct_dir = os.path.join(root_dir, 'instructions')
    
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(control_dir, exist_ok=True)
    os.makedirs(instruct_dir, exist_ok=True)
    
    for i in range(num_samples):
        stem = f"sample_{i:03d}"
        # Create dummy images
        Image.new('RGB', (64, 64), color='red').save(os.path.join(img_dir, f"{stem}.jpg"))
        Image.new('RGB', (64, 64), color='blue').save(os.path.join(control_dir, f"{stem}.jpg"))
        
        # Create dummy text
        with open(os.path.join(instruct_dir, f"{stem}.txt"), 'w') as f:
            f.write(f"instruction for {stem}")
            
    return img_dir, control_dir, instruct_dir

def encode_latents_mock(img):
    # Return a dummy tensor
    return torch.randn(4, 32, 32)

def encode_prompt_mock(text):
    # Return a dummy tensor
    return torch.randn(77, 768)

def run_test():
    # Initialize process group
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    print(f"Rank {rank}/{world_size} started.")
    
    # Use a shared temporary directory for all processes
    # In a real distributed setting, this should be a shared filesystem
    # Here we assume local execution
    base_dir = "/tmp/meica_dist_test"
    
    if rank == 0:
        if os.path.exists(base_dir):
            shutil.rmtree(base_dir)
        os.makedirs(base_dir, exist_ok=True)
        # Use 21 samples to test uneven distribution with 2 ranks (11 vs 10)
        num_samples = 21
        img_dir, ctrl_dir, inst_dir = setup_dummy_data(base_dir, num_samples=num_samples)
        print(f"Rank 0 created dummy data at {base_dir} with {num_samples} samples")
    
    dist.barrier()
    
    # Need to define num_samples for other ranks or just trust the filesystem scan
    # For assertions, we'll hardcode 21
    total_expected = 21
    
    img_dir = os.path.join(base_dir, 'images')
    ctrl_dir = os.path.join(base_dir, 'controls')
    inst_dir = os.path.join(base_dir, 'instructions')
    cache_dir = os.path.join(base_dir, 'cache')
    binary_dir = os.path.join(base_dir, 'binary_output')
    
    # Test 1: Distributed Precomputation
    print(f"[Rank {rank}] Starting Dataset Initialization (Precompute)...")
    dataset = ImageToImageDataset(
        img_dir=img_dir,
        control_img_dir=ctrl_dir,
        instruction_dir=inst_dir,
        precompute_embeddings=True,
        encode_latents=encode_latents_mock,
        encode_prompt=encode_prompt_mock,
        cache_dir=cache_dir
    )
    
    # Verify cache files exist for assigned portion
    
    dist.barrier()
    
    if rank == 0:
        # Check total cache files
        cached_files = list(Path(cache_dir).glob("*_embed.pt"))
        print(f"Total cached embeddings found: {len(cached_files)}")
        assert len(cached_files) == total_expected, f"Expected {total_expected} cached embeddings, found {len(cached_files)}"
        print("Test 1 Passed: Distributed Precomputation complete.")

    dist.barrier()
    
    # Test 2: Distributed Binary Export
    print(f"[Rank {rank}] Starting Binary Export...")
    dataset.export_to_binary(binary_dir, chunk_size=1024*1024) # 1MB chunk
    
    dist.barrier()
    
    if rank == 0:
        # Check index file
        index_path = os.path.join(binary_dir, "index.bin")
        assert os.path.exists(index_path), "Index file not found"
        
        index_records = BinaryIndex.load(index_path)
        print(f"Total binary records exported: {len(index_records)}")
        assert len(index_records) == total_expected, f"Expected {total_expected} binary records, found {len(index_records)}"
        
        # Check chunk files
        chunk_files = list(Path(binary_dir).glob("chunk_*.bin"))
        print(f"Generated chunks: {[p.name for p in chunk_files]}")
        assert len(chunk_files) >= 1, "No chunk files found"
        
        # Verify data integrity of a random sample
        dataset_binary = ImageToImageDataset(
            img_dir="", control_img_dir="", instruction_dir="", # Not needed for binary
            binary_path=binary_dir
        )
        sample = dataset_binary[total_expected - 1] # Check the last sample
        print(f"Loaded sample {total_expected - 1} from binary: {sample.keys()}")
        assert 'embedding' in sample
        assert 'latent' in sample
        
        print("Test 2 Passed: Distributed Binary Export complete.")
        
        # Cleanup
        shutil.rmtree(base_dir)

    dist.destroy_process_group()

if __name__ == "__main__":
    # Ensure we are running with torchrun
    if "RANK" not in os.environ:
        print("Please run this script using torchrun:\n torchrun --nproc_per_node=2 example/test_dist_dataset.py")
    else:
        run_test()
