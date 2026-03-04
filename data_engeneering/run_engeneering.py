import xarray as xr
import glob
import os
from merging import sanitize_netcdf
from engeneering import engineer_physical_features

def process_features():
    output_dir = "./data/processed/"
    os.makedirs(output_dir, exist_ok=True)
    
    # Verify the target raw data directory exists before starting the loop
    raw_dir = "./data/era5_france_raw"
    if not os.path.exists(raw_dir):
        raise FileNotFoundError(f"CRITICAL: The directory '{raw_dir}' does not exist in the current working directory: {os.getcwd()}")
    
    for year in range(1996, 2026):
        print(f"\nEvaluating year: {year}...")
        
        surf_pattern = f"{raw_dir}/surface_{year}_*.nc"
        pres_pattern = f"{raw_dir}/pressure_{year}_*.nc"
        
        surf_files = glob.glob(surf_pattern)
        pres_files = glob.glob(pres_pattern)
        
        # Explicit Logging 
        print(f"  -> Target surface pattern: {surf_pattern}")
        print(f"  -> Found {len(surf_files)} surface files.")
        print(f"  -> Found {len(pres_files)} pressure files.")
        
        if not surf_files or not pres_files:
            print(f"  -> WARNING: Missing data for {year}. Skipping to next year.")
            continue
            
        surf_files.sort()
        pres_files.sort()
        
        print("  -> Loading lazy coordinate matrices...")
        ds_surf = xr.open_mfdataset(
            surf_files, combine='by_coords', 
            preprocess=sanitize_netcdf, chunks={'valid_time': 'auto'}
        )
        ds_pres = xr.open_mfdataset(
            pres_files, combine='by_coords', 
            preprocess=sanitize_netcdf, chunks={'valid_time': 'auto'}
        )

        ds_surf, ds_pres = xr.align(ds_surf, ds_pres, join='exact')
        print("  -> Engineering physical feature graphs...")
        ds_eng = engineer_physical_features(ds_surf, ds_pres)
        
        print("  -> Flattening vertical pressure levels into discrete spatial channels...")
        # Initialize a strictly 3D container (Time, Latitude, Longitude)
        ds_pres_flat = xr.Dataset(coords={
            'valid_time': ds_pres.valid_time, 
            'latitude': ds_pres.latitude, 
            'longitude': ds_pres.longitude
        })
        
        # Project each variable/level combination into a distinct flat channel
        for var in ds_pres.data_vars:
            for lvl in ds_pres.pressure_level.values:
                channel_name = f"{var}_{int(lvl)}"
                # Isolate the level and explicitly drop the Z-coordinate metadata
                ds_pres_flat[channel_name] = ds_pres[var].sel(pressure_level=lvl).drop_vars('pressure_level')
                
        print("  -> Merging strictly 3D state matrices with engineered features...")
        # ds_unified is now guaranteed to be strictly R^(T x 64 x 64) across all variables
        ds_surf, ds_pres_flat, ds_eng = xr.align(ds_surf, ds_pres_flat, ds_eng, join='exact')
        ds_unified = xr.merge([ds_surf, ds_pres_flat, ds_eng])
        
        output_path = os.path.join(output_dir, f"engineered_{year}.nc")
        print(f"  -> Executing compute graph and serializing to {output_path}...")
        
        # Serialize the complete 95-channel flattened dataset
        ds_unified.to_netcdf(output_path)
        print("  -> Success.")

if __name__ == "__main__":
    process_features()