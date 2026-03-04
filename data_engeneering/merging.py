import xarray as xr
import glob

def sanitize_netcdf(ds):
    """
    Mathematical interception function.
    Executes on each file independently before the concatenation phase.
    """
    ds.attrs.clear()
    
    if 'expver' in ds.variables or 'expver' in ds.dims:
        ds = ds.isel(expver=0, drop=True)
        
    return ds

# Wrap the testing execution!
if __name__ == "__main__":
    surface_files = glob.glob("./era5_france_raw/surface_*.nc")
    surface_files.sort()

    print(f"Detected {len(surface_files)} surface fragments. Initiating merge...")

    ds_surface = xr.open_mfdataset(
        surface_files, 
        combine='by_coords',
        preprocess=sanitize_netcdf,
        parallel=True,
        chunks={'valid_time': 'auto'}
    )

    print("\nUnified Surface Channel Structure:")
    print(list(ds_surface.data_vars.keys()))