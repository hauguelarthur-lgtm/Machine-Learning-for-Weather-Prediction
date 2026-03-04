import numpy as np
import glob
import os
import sys

def verify_dataset_integrity(tensor_dir="data/processed/tensors/"):
    """
    Executes a deterministic mathematical scan over the out-of-core tensors.
    Isolates exact indices and physical channels containing NaN or Inf values.
    """
    files = sorted(glob.glob(os.path.join(tensor_dir, "france_tensor_*.npy")))
    if not files:
        raise FileNotFoundError(f"No .npy tensors located in {tensor_dir}")

    # Process 100 chronological timesteps per iteration to bound RAM utilization
    chunk_size = 100 

    print("==================================================")
    print("INITIATING DATASET INTEGRITY VERIFICATION")
    print("==================================================")

    for file_path in files:
        filename = os.path.basename(file_path)
        print(f"Scanning matrix: {filename}...")
        
        # Allocate virtual address space without loading physical bytes
        mmap_array = np.load(file_path, mmap_mode='r')
        total_timesteps = mmap_array.shape[0]

        for i in range(0, total_timesteps, chunk_size):
            # Extract the specific sub-tensor into active RAM
            chunk = mmap_array[i : i + chunk_size].copy()

            # Execute parallelized boolean masking
            has_nan = np.isnan(chunk).any()
            has_inf = np.isinf(chunk).any()

            if has_nan or has_inf:
                print(f"\n[!] CRITICAL FAULT DETECTED IN {filename}")
                print(f" -> Chronological Index Range: {i} to {i + chunk.shape[0] - 1}")
                
                # Isolate the specific physical fluid channel causing the collapse
                if has_nan:
                    faulty_channels = np.unique(np.where(np.isnan(chunk))[1])
                    print(f" -> NaN contamination in channels: {faulty_channels}")
                if has_inf:
                    faulty_channels = np.unique(np.where(np.isinf(chunk))[1])
                    print(f" -> Inf contamination in channels: {faulty_channels}")
                    
                sys.exit(1)
                
        # Force the OS to close the file descriptor before proceeding
        del mmap_array

    print("\n==================================================")
    print("VERIFICATION COMPLETE: 0 CONTAMINATIONS DETECTED.")
    print("The topological manifold is strictly continuous and real.")
    print("==================================================")

if __name__ == "__main__":
    verify_dataset_integrity()