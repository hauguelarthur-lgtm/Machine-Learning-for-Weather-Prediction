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
    # Create array from -30 to +30
    offsets = np.arange(-radius, radius + 1)
    
    # Calculate linear decay: 1 - (|k| / 31)
    weights = np.maximum(0, 1 - (np.abs(offsets) / (radius + 1)))
    
    # Normalize the kernel to maintain physical magnitudes
    return xr.DataArray(weights / weights.sum(), dims=['window'])

def compute_wb2_baseline():
    output_dir = "./data/processed/evaluation/"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Locating 1990-2019 baseline ERA5 matrices...")
    # NOTE: You must only execute this on the specific variables you intend to evaluate
    # (e.g., Z500, T850, T2M) to minimize SSD I/O.
    # Assuming you flattened the target fields into specific .nc files for this phase:
    target_files = sorted(glob.glob("./era5_france_raw/evaluation_targets_1990_2019*.nc"))
    
    if not target_files:
        raise FileNotFoundError("Missing 1990-2019 target variables. Download required.")

    ds = xr.open_mfdataset(target_files, chunks={'valid_time': 'auto'})
    
    print("Calculating raw temporal means across the 30-year axis...")
    # Group by Day of Year (1 to 366) and Hour (0, 6, 12, 18)
    # This collapses the T dimension from ~43,800 to exactly 1464
    raw_climatology = ds.groupby('valid_time.dayofyear').mean('valid_time')
    
    print("Applying cyclical 61-day convolution kernel...")
    kernel = create_triangular_kernel(window_size=61)
    
    # To compute the rolling window across the Dec 31 -> Jan 1 boundary, 
    # we must pad the dataset cyclically along the dayofyear dimension.
    pad_width = 30
    padded_climatology = raw_climatology.pad(dayofyear=pad_width, mode='wrap')
    
    # Construct the Dask rolling graph
    # construct() exposes the rolling window as a new dimension, 
    # allowing direct element-wise multiplication with our custom kernel
    rolling_obj = padded_climatology.rolling(dayofyear=61, center=True).construct('window')
    
    # Execute the dot product along the window dimension
    smoothed_climatology = (rolling_obj * kernel).sum('window', skipna=False)
    
    # Strip the padding to return to the strict 366-day matrix
    final_climatology = smoothed_climatology.isel(
        dayofyear=slice(pad_width, -pad_width)
    )
    
    output_path = os.path.join(output_dir, "wb2_climatology_baseline.nc")
    print(f"Serializing mathematical baseline to {output_path}...")
    
    # Trigger the optimized compute graph
    final_climatology.to_netcdf(output_path)
    print("Success. WB2 Climatological baseline established.")

if __name__ == "__main__":
    compute_wb2_baseline()