import xarray as xr
import glob

def sanitize_netcdf(ds):
    """
    Mathematical interception function.
    Executes on each file independently before the concatenation phase.
    """
    # 1. Erase the conflicting global cfgrib history dictionaries
    ds.attrs.clear()
    
    # 2. Drop the 'expver' (Experiment Version) array. 
    # This non-physical string array disrupts coordinate alignment.
    if 'expver' in ds.variables:
        ds = ds.drop_vars('expver')
        
    return ds

# Programmatically gather the localized fragments
surface_files = glob.glob("./era5_france_raw/surface_*.nc")
surface_files.sort()

print(f"Detected {len(surface_files)} surface fragments. Initiating merge...")

# Construct the unified dataset dynamically
ds_surface = xr.open_mfdataset(
    surface_files, 
    combine='by_coords',
    preprocess=sanitize_netcdf,    # Cleans the tensors before merging
    parallel=True,                 # Distributes the load across CPU cores
    chunks={'valid_time': 'auto'}  # Aligns RAM allocation with disk sectors
)

print("\nUnified Surface Channel Structure:")
print(list(ds_surface.data_vars.keys()))