import xarray as xr
import numpy as np
import json
import os
import gc

def encode_static_variables(ds_block):
    """Transforms static variables for optimal neural network ingestion."""
    if 'lsm' in ds_block.data_vars:
        ds_block['lsm_centered'] = (ds_block['lsm'] * 2.0) - 1.0
        ds_block = ds_block.drop_vars('lsm')

    if 'slt' in ds_block.data_vars:
        valid_soil_classes = [0, 1, 2, 3, 4, 5, 6, 7] 
        for soil_class in valid_soil_classes:
            channel_name = f"slt_class_{soil_class}"
            ds_block[channel_name] = (ds_block['slt'] == soil_class).astype(np.float32)
        ds_block = ds_block.drop_vars('slt')

    return ds_block

def process_and_serialize_chunks():
    # 1. Load the Global Statistics
    with open("./data/processed/global_stats.json", "r") as f:
        stats = json.load(f)
    
    mu = xr.Dataset({k: xr.DataArray(v) for k, v in stats["mean"].items()})
    sigma = xr.Dataset({k: xr.DataArray(v) for k, v in stats["std"].items()})
    EPSILON = 1e-8 
    
    blocks = [
        ("1996", "2000"), ("2001", "2005"), ("2006", "2010"),
        ("2011", "2015"), ("2016", "2020"), ("2021", "2025")
    ]
    
    continuous_vars = list(stats["mean"].keys())
    encoded_static_vars = [
        'lsm_centered', 
        'slt_class_0', 'slt_class_1', 'slt_class_2', 'slt_class_3', 
        'slt_class_4', 'slt_class_5', 'slt_class_6', 'slt_class_7',
        'hour_sin', 'hour_cos', 'day_sin', 'day_cos'
    ]
    
    channel_order = continuous_vars + encoded_static_vars
    os.makedirs("./data/processed/tensors/", exist_ok=True)
    with open("./data/processed/tensors/channel_ordering.json", "w") as f:
        json.dump(channel_order, f, indent=4)
    print(f"Locked physical channel order serialized to JSON. Total channels: {len(channel_order)}")
    
    # 2. Isolated Iteration Protocol
    for start_year, end_year in blocks:
        print(f"\nProcessing block: {start_year} - {end_year}")
        
        # Dynamically define only the files needed for this specific 5-year chunk
        block_files = [
            f"./data/processed/engineered_{year}.nc" 
            for year in range(int(start_year), int(end_year) + 1)
        ]
        
        # Verify physical files exist to prevent glob failures
        valid_files = [f for f in block_files if os.path.exists(f)]
        if not valid_files:
            print(f"  -> WARNING: No files found for {start_year}-{end_year}. Skipping.")
            continue
            
        # Open a fresh, localized Dask graph
        ds_block = xr.open_mfdataset(valid_files, chunks={'valid_time': 'auto'})
        
        # Execute mathematical graph definitions
        ds_block = encode_static_variables(ds_block)
        ds_cont = ds_block[continuous_vars]
        ds_norm = (ds_cont - mu) / (sigma + EPSILON)
        
        da_norm = ds_norm[continuous_vars].to_array(dim='channel')
        da_excl = ds_block[encoded_static_vars].to_array(dim='channel')
        
        tensor_da = xr.concat([da_norm, da_excl], dim='channel')
        tensor_da = tensor_da.transpose('valid_time', 'channel', 'latitude', 'longitude')
        
        # Trigger computation 
        print(f"  -> Executing compute graph for {start_year}-{end_year} (~6.5 GB optimized RAM spike)...")
        numpy_tensor = tensor_da.astype(np.float32).values
        
        output_path = f"./data/processed/tensors/france_tensor_{start_year}_{end_year}.npy"
        np.save(output_path, numpy_tensor)
        print(f"  -> Saved {output_path} with shape {numpy_tensor.shape}")

        # 3. Deep Graph Cleansing
        ds_block.close()  # Force OS to release file handles
        
        # Erase all intermediate graph components from Python's namespace
        del numpy_tensor
        del tensor_da
        del da_norm
        del da_excl
        del ds_norm
        del ds_cont
        del ds_block
        
        # Run garbage collection
        gc.collect()

if __name__ == "__main__":
    process_and_serialize_chunks()