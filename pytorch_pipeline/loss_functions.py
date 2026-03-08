import torch
import torch.nn as nn
import numpy as np

class LatitudeWeightedMSELoss(nn.Module):
    def __init__(self, latitudes):
        """
        Initializes the latitude-weighted Mean Squared Error.
        
        Parameters:
        latitudes (np.ndarray or list): The 1D array of latitude coordinates.
                                        For this specific grid, length = 64.
        """
        super().__init__()
        
        # 1. Mathematical Transformation
        # Convert degrees to radians and extract the cosine
        latitudes_rad = np.deg2rad(np.array(latitudes, dtype=np.float32))
        weights = np.cos(latitudes_rad)
        
        # 2. Normalization
        # Mean scaling ensures the global learning rate is not artificially suppressed.
        # If the weights sum to a value significantly different from N, the 
        # optimizer gradients will vanish or explode.
        weights_normalized = weights / np.mean(weights)
        
        # 3. Broadcasting Alignment
        # The PyTorch output tensor is R^(Batch, Channel, Height, Width).
        # Height represents the latitude axis (index 2). 
        # We must reshape the 1D weight array to (1, 1, 64, 1) to allow 
        # vectorized Hadamard multiplication.
        weight_tensor = torch.tensor(weights_normalized).view(1, 1, len(latitudes), 1)
        
        # 4. Hardware Registration
        # register_buffer stores the tensor inside the nn.Module state_dict, 
        # but excludes it from the autograd graph (it does not require gradients).
        # When model.to('cuda') is called, this matrix automatically moves to the GPU.
        self.register_buffer('weight_matrix', weight_tensor)

    def forward(self, predictions, targets):
        """
        Executes the weighted derivative evaluation.
        predictions, targets: R^(B x 102 x 64 x 64)
        """
        # Calculate standard pixel-wise squared error
        squared_error = (predictions - targets) ** 2
        
        # Apply the broadcasted geographical weighting
        weighted_squared_error = squared_error * self.weight_matrix
        
        # Reduce to a scalar for the backward pass
        return torch.mean(weighted_squared_error)

# Execution Test & Latitude Extraction
if __name__ == "__main__":
    import xarray as xr
    
    # Extract the exact latitude matrix directly from the raw data
    # to guarantee coordinate alignment.
    try:
        sample_ds = xr.open_dataset("../era5_france_raw/surface_1996_01_part1.nc")
        lats = sample_ds['latitude'].values
        print(f"Extracted Latitudes: {lats[0]} to {lats[-1]}, Shape: {lats.shape}")
        
        criterion = LatitudeWeightedMSELoss(latitudes=lats)
        
        # Simulate a forward pass
        dummy_pred = torch.randn(16, 102, 64, 64)
        dummy_targ = torch.randn(16, 102, 64, 64)
        
        loss = criterion(dummy_pred, dummy_targ)
        print(f"Computed Weighted Loss: {loss.item():.4f}")
        
    except FileNotFoundError:
        print("Raw NetCDF file not found. Ensure path is correct for execution test.")