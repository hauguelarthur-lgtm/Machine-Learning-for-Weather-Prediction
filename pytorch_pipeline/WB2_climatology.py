import xarray as xr
import numpy as np
import os
import glob

def create_triangular_kernel(window_size=61):
    """
    Generates the strict WB2 linearly decaying weight matrix.
    A window of 61 yields a radius of 30 days.
    """
    radius = window_size // 2
    offsets = np.arange(-radius, radius + 1)
    
    # Calculate linear decay: 1 - (|k| / 31)
    weights = np.maximum(0, 1 - (np.abs(offsets) / (radius + 1)))
    return xr.DataArray(weights / weights.sum(), dims=['window'])

def compute_wb2_baseline():
    output_dir = "./data/processed/evaluation/"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Locating 1990-2019 baseline ERA5 matrices...")
    target_files = sorted(glob.glob("./data/processed/engineered*.nc"))
    
    if not target_files:
        raise FileNotFoundError("Missing 1990-2019 target variables. Download required.")

    ds = xr.open_mfdataset(target_files, chunks={'valid_time': 'auto'})
    
    # 3. Memory Isolation Matrix
    # Strictly define the variables targeted by the WeatherBench 2 evaluation.
    eval_vars = ['z_500', 't_850', 't2m', 'u10', 'v10'] 
    
    # Slicing the graph strictly prevents Dask from loading the other 97 channels into RAM
    ds = ds[eval_vars]

    print("Extracting diurnal cycles and calculating raw 30-year temporal means...")
    # 1. Execute a 2D mathematical reduction to preserve the solar cycle boundaries
    def mean_by_day(ds_hour_block):
        return ds_hour_block.groupby('valid_time.dayofyear').mean('valid_time')
        
    # Maps across hours, then reduces across days. Output is R^(hour x dayofyear x lat x lon)
    raw_climatology = ds.groupby('valid_time.hour').map(mean_by_day)
    
    print("Applying cyclical 61-day convolution kernel exclusively across the dayofyear axis...")
    kernel = create_triangular_kernel(window_size=61)
    
    # Pad the dayofyear boundary cyclically (Dec 31 wraps to Jan 1)
    pad_width = 30
    padded_climatology = raw_climatology.pad(dayofyear=pad_width, mode='wrap')
    
    # Evaluate the structural dot product strictly along the window dimension
    rolling_obj = padded_climatology.rolling(dayofyear=61, center=True).construct('window')
    smoothed_climatology = (rolling_obj * kernel).sum('window', skipna=False)
    
    # Strip the padding to return to the strict 366-day manifold
    final_climatology = smoothed_climatology.isel(dayofyear=slice(pad_width, -pad_width))
    
    print("Formatting spatial baseline for O(1) inference evaluation lookup...")
    # 1. Collapse the 2D temporal axes into a temporary MultiIndex
    final_climatology = final_climatology.stack(temp_dim=['dayofyear', 'hour'])
    
    # 2. Generate the strict "DDD_HH" string list
    doy_hour_strings = [f"{int(d):03d}_{int(h):02d}" for d, h in final_climatology['temp_dim'].values]
    
    # 3. Assign the strings as a new independent coordinate along the temporary dimension
    final_climatology = final_climatology.assign_coords(doy_hour=("temp_dim", doy_hour_strings))
    
    # 4. Swap the topological dimension pointer to the new string coordinate
    final_climatology = final_climatology.swap_dims({"temp_dim": "doy_hour"})
    
    # 5. Erase the legacy MultiIndex and its sub-components to allow NetCDF4 serialization
    final_climatology = final_climatology.drop_vars(['temp_dim', 'dayofyear', 'hour'], errors='ignore')
    
    output_path = os.path.join(output_dir, "wb2_climatology_baseline.nc")
    print(f"Serializing mathematical baseline to {output_path}...")
    
    # Trigger Dask reduction into VRAM/RAM and flush to NVMe
    final_climatology.to_netcdf(output_path)
    print("Success. WB2 Climatological baseline established.")

if __name__ == "__main__":
    compute_wb2_baseline()