import xarray as xr
import glob

# 1. Programmatically gather all surface fragments
# This dynamically catches files whether they have a '_partX' suffix or not
surface_files = glob.glob("./era5_france_raw/surface_*.nc")
surface_files.sort()

print(f"Detected {len(surface_files)} surface fragments. Initiating merge...")

# 2. Construct the unified dataset
ds_surface = xr.open_mfdataset(
    surface_files, 
    combine='by_coords',     # Aligns the physical (lat/lon) and temporal coordinates
    compat='override',       # Bypasses the ECMWF metadata string conflicts
    data_vars='minimal',     # Prevents coordinate variables from being duplicated as data
    parallel=True,           # Utilizes dask to load the chunks across your 8 CPU cores
    chunks={'valid_time': 120} 
)

# Verify the unified variable extraction
print("\nUnified Surface Channel Structure:")
print(list(ds_surface.data_vars.keys()))