import torch
import torch.nn as nn
import numpy as np

class WeatherBench2Metrics:
    def __init__(self, latitudes):
        """
        Initializes the WB2 evaluation metrics.
        latitudes: 1D array of latitude coordinates.
        """
        lat_rad = np.deg2rad(np.array(latitudes, dtype=np.float32))
        weights = np.cos(lat_rad)
        weights_normalized = weights / np.mean(weights)
        
        # Shape: (1, 1, Latitude, 1) for broadcasting over (Batch, Channel, Lat, Lon)
        self.weights = torch.tensor(weights_normalized).view(1, 1, len(latitudes), 1)

    def to(self, device):
        """Moves the weight matrix to the target GPU."""
        self.weights = self.weights.to(device)

    def compute_rmse(self, predictions, targets):
        """
        Computes the area-weighted RMSE per channel.
        WB2 Equation: sqrt( 1/TIJ * sum( w(i) * (f - o)^2 ) )
        """
        # 1. Calculate squared error
        sq_error = (predictions - targets) ** 2
        
        # 2. Apply latitude weighting
        weighted_sq_error = sq_error * self.weights
        
        # 3. Mean across Batch (T), Latitude (I), and Longitude (J) dimensions
        # Returns a 1D tensor of length C (one RMSE value per physical channel)
        mean_weighted_sq_error = torch.mean(weighted_sq_error, dim=(0, 2, 3))
        
        # 4. Square root evaluated AFTER the temporal/spatial mean
        rmse = torch.sqrt(mean_weighted_sq_error)
        return rmse

    def compute_wind_vector_rmse(self, u_pred, v_pred, u_target, v_target):
        """
        Computes the Wind Vector RMSE for specific pressure levels.
        WB2 Equation: sqrt( 1/TIJ * sum( w(i) * [ (u_f - u_o)^2 + (v_f - v_o)^2 ] ) )
        """
        sq_error_u = (u_pred - u_target) ** 2
        sq_error_v = (v_pred - v_target) ** 2
        
        vector_error = sq_error_u + sq_error_v
        weighted_vector_error = vector_error * self.weights
        
        mean_weighted_error = torch.mean(weighted_vector_error, dim=(0, 2, 3))
        return torch.sqrt(mean_weighted_error)

    def compute_bias(self, predictions, targets):
        """
        Computes the global spatial mean error (Bias).
        WB2 Equation: 1/TIJ * sum( w(i) * (f - o) )
        """
        error = predictions - targets
        weighted_error = error * self.weights
        
        # Returns a 1D tensor of length C
        return torch.mean(weighted_error, dim=(0, 2, 3))