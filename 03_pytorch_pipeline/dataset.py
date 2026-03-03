import torch
from torch.utils.data import Dataset
import numpy as np
import glob
import bisect
import os

class MeteorologicalDataset(Dataset):
    def __init__(self, tensor_dir="./data/processed/tensors/", horizon=1):
        """
        Initializes the out-of-core memory mapped dataset.
        horizon: The autoregressive predictive gap. 
                 If data is 6-hourly, horizon=1 predicts +6h, horizon=4 predicts +24h.
        """
        self.horizon = horizon
        self.tensor_files = sorted(glob.glob(os.path.join(tensor_dir, "france_tensor_*.npy")))
        
        if not self.tensor_files:
            raise FileNotFoundError(f"No serialized .npy tensors found in {tensor_dir}")

        self.mmaps = []
        self.cumulative_sizes = [0]
        
        # Initialize memory maps and calculate global tensor boundaries
        for file_path in self.tensor_files:
            # mmap_mode='r' parses the .npy header and allocates the virtual address space
            # without pulling the physical bytes into active RAM.
            mmap_array = np.load(file_path, mmap_mode='r')
            self.mmaps.append(mmap_array)
            
            # The temporal dimension is strictly axis 0
            T_k = mmap_array.shape[0]
            self.cumulative_sizes.append(self.cumulative_sizes[-1] + T_k)
            
        self.total_timesteps = self.cumulative_sizes[-1]
        
        # The dataset must stop early enough to allow the final X to reach the final Y
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
        """
        Extracts the input tensor X and the target tensor Y directly from the SSD.
        """
        # 1. Resolve physical location for state X
        file_X, local_X = self._resolve_global_index(idx)
        
        # 2. Resolve physical location for target state Y
        target_idx = idx + self.horizon
        file_Y, local_Y = self._resolve_global_index(target_idx)
        
        # 3. Read specific contiguous byte blocks from SSD
        # copy() is required to force NumPy to convert the read-only memory map reference
        # into a mutable array in active RAM, which PyTorch requires for gradient attachment.
        x_array = self.mmaps[file_X][local_X].copy()
        y_array = self.mmaps[file_Y][local_Y].copy()
        
        # 4. Convert to PyTorch tensors (C, H, W)
        x_tensor = torch.from_numpy(x_array)
        y_tensor = torch.from_numpy(y_array)
        
        return x_tensor, y_tensor

# Execution Test
if __name__ == "__main__":
    dataset = MeteorologicalDataset(horizon=1)
    print(f"Total valid training pairs: {len(dataset)}")
    
    # Extract a batch across a file boundary to verify cross-mapping logic
    x, y = dataset[7307] 
    print(f"X shape: {x.shape}")
    print(f"Y shape: {y.shape}")