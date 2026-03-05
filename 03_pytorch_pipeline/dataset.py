import torch
from torch.utils.data import Dataset
import numpy as np
import glob
import bisect
import os


class MeteorologicalDataset(Dataset):
    def __init__(self, tensor_dir="./data/processed/tensors/", rollout_steps=3):
        self.rollout_steps = rollout_steps
        self.tensor_files = sorted(glob.glob(os.path.join(tensor_dir, "france_tensor_*.npy")))
        
        self.mmaps = None 
        self.cumulative_sizes = [0]
        
        # Parse headers to build the boundary index
        for file_path in self.tensor_files:
            mmap_array = np.load(file_path, mmap_mode='r')
            T_k = mmap_array.shape[0]
            self.cumulative_sizes.append(self.cumulative_sizes[-1] + T_k)
            del mmap_array
            
        self.total_timesteps = self.cumulative_sizes[-1]
        self.valid_length = self.total_timesteps - self.rollout_steps

    def _resolve_global_index(self, idx):
        file_idx = bisect.bisect_right(self.cumulative_sizes, idx) - 1
        local_idx = idx - self.cumulative_sizes[file_idx]
        return file_idx, local_idx

    def __len__(self):
        return self.valid_length

    def __getitem__(self, idx):
        if self.mmaps is None:
            self.mmaps = [np.load(f, mmap_mode='r') for f in self.tensor_files]
            
        file_X, local_X = self._resolve_global_index(idx)
        x_array = self.mmaps[file_X][local_X].copy()
        
        # Extract the continuous fluid trajectory: R^(K x 102 x 64 x 64)
        y_sequence = []
        for k in range(1, self.rollout_steps + 1):
            file_Y, local_Y = self._resolve_global_index(idx + k)
            y_sequence.append(torch.from_numpy(self.mmaps[file_Y][local_Y].copy()))
            
        return torch.from_numpy(x_array), torch.stack(y_sequence, dim=0)



if __name__ == "__main__":
    dataset = MeteorologicalDataset(rollout_steps=3)
    print(f"Total valid training pairs: {len(dataset)}")
    x, y = dataset[7307] 
    print(f"X shape: {x.shape}")
    print(f"Y Sequence shape: {y.shape}")