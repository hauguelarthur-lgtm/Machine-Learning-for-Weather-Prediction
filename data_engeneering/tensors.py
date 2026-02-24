import xarray as xr
import numpy as np
import json
import os

def process_and_serialize_chunks():
    # 1. Load the Global Statistics
    with open("./data/processed/global_stats.json", "r") as f:
        stats = json.load(f)
    
    mu = xr.Dataset({k: xr.DataArray(v) for k, v in stats["mean"].items()})
    sigma = xr.Dataset({k: xr.DataArray(v) for k, v in stats["std"].items()})
    
    # Mathematical Epsilon to prevent division by zero in zero-variance fields
    EPSILON = 1e-8 
    
    # 2. Define the 5-Year Temporal Blocks
    blocks = [
        ("1996", "2000"), ("2001", "2005"), ("2006", "2010"),
        ("2011", "2015"), ("2016", "2020"), ("2021", "2025")
    ]
    
    ds = xr.open_mfdataset("./data/processed/engineered_*.nc", chunks={'valid_time': 'auto'})
    
    excluded_vars = ['lsm', 'slt', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos']
    continuous_vars = list(stats["mean"].keys())
    
    os.makedirs("./data/processed/tensors/", exist_ok=True)
    
    # 3. Iterative Execution
    for start_year, end_year in blocks:
        print(f"Processing block: {start_year} - {end_year}")
        
        # Slice the lazy graph
        ds_block = ds.sel(valid_time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))
        
        # A. Isolate and Normalize Continuous Variables
        ds_cont = ds_block[continuous_vars]
        ds_norm = (ds_cont - mu) / (sigma + EPSILON)
        
        # Convert to xarray DataArray with a new 'channel' dimension
        da_norm = ds_norm.to_array(dim='channel')
        
        # B. Isolate Static and Cyclical Variables (No Normalization)
        ds_excl = ds_block[excluded_vars]
        da_excl = ds_excl.to_array(dim='channel')
        
        # C. Concatenate along the channel dimension (C = 53)
        tensor_da = xr.concat([da_norm, da_excl], dim='channel')
        
        # D. Transpose to strict PyTorch shape: (Time, Channel, Height, Width)
        tensor_da = tensor_da.transpose('valid_time', 'channel', 'latitude', 'longitude')
        
        # E. Compute and Serialize
        # This triggers the Dask evaluation for this specific 5-year chunk
        print(f"  -> Executing compute graph for {start_year}-{end_year} (~6.34 GB RAM spike)...")
        numpy_tensor = tensor_da.values.astype(np.float32)
        
        output_path = f"./data/processed/tensors/france_tensor_{start_year}_{end_year}.npy"
        np.save(output_path, numpy_tensor)
        print(f"  -> Saved {output_path} with shape {numpy_tensor.shape}")

        # Explicitly delete the array to free RAM for the next iteration
        del numpy_tensor
        del tensor_da

if __name__ == "__main__":
    process_and_serialize_chunks()