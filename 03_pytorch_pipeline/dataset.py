import torch
from torch.utils.data import Dataset
import numpy as np
import glob
import bisect
import os

class MeteorologicalDataset(Dataset):
    def __init__(self, tensor_dir="./data/processed/tensors/", horizon=1):
        self.horizon = horizon
        self.tensor_files = sorted(glob.glob(os.path.join(tensor_dir, "france_tensor_*.npy")))
        
        self.mmaps = None # Strictly uninitialized in the parent process
        self.cumulative_sizes = [0]
        
        # Parse headers to build the boundary index, but close the descriptor immediately
        for file_path in self.tensor_files:
            mmap_array = np.load(file_path, mmap_mode='r')
            T_k = mmap_array.shape[0]
            self.cumulative_sizes.append(self.cumulative_sizes[-1] + T_k)
            # Delete the reference to force the OS to release the file descriptor
            del mmap_array
            
        self.total_timesteps = self.cumulative_sizes[-1]
        self.valid_length = self.total_timesteps - self.horizon

    def _resolve_global_index(self, idx):
        """
        Maps a global temporal index to the specific physical file and local index.
        Utilizes binary search (bisect) for O(log K) routing efficiency.
        """
        # bisect_right finds the insertion point. Subtract 1 to get the file index k.
        file_idx = bisect.bisect_right(self.cumulative_sizes, idx) - 1
        
        # Calculate local offset
        local_idx = idx - self.cumulative_sizes[file_idx]
        
        return file_idx, local_idx

    def __len__(self):
        """Returns the total number of valid (X, Y) training pairs."""
        return self.valid_length

    def __getitem__(self, idx):
        # Lazy initialization: The worker process opens its own isolated file descriptors
        if self.mmaps is None:
            self.mmaps = [np.load(f, mmap_mode='r') for f in self.tensor_files]
            
        file_X, local_X = self._resolve_global_index(idx)
        target_idx = idx + self.horizon
        file_Y, local_Y = self._resolve_global_index(target_idx)
        
        x_array = self.mmaps[file_X][local_X].copy()
        y_array = self.mmaps[file_Y][local_Y].copy()
        
        return torch.from_numpy(x_array), torch.from_numpy(y_array)

# Execution Test
if __name__ == "__main__":
    dataset = MeteorologicalDataset(horizon=1)
    print(f"Total valid training pairs: {len(dataset)}")
    
    # Extract a batch across a file boundary to verify cross-mapping logic
    x, y = dataset[7307] 
    print(f"X shape: {x.shape}")
    print(f"Y shape: {y.shape}")