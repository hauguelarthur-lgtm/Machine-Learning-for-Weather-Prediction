import xarray as xr
import numpy as np
import glob
import os

def verify_engineered_files(processed_dir="./data/processed/"):
    """
    Executes a deterministic validation of the engineered NetCDF files.
    Evaluates matrix dimensions, temporal contiguity, and mathematical validity.
    """
    files = sorted(glob.glob(os.path.join(processed_dir, "engineered_*.nc")))
    if not files:
        raise FileNotFoundError(f"No engineered files located in {processed_dir}")

    print("==================================================")
    print("INITIATING ENGINEERED NetCDF VERIFICATION")
    print("==================================================")

    for file_path in files:
        filename = os.path.basename(file_path)
        print(f"\nScanning: {filename}")
        
        try:
            # Lazy load the dataset
            ds = xr.open_dataset(file_path)
            
            # 1. Dimensional Topology Check
            expected_dims = {'valid_time', 'latitude', 'longitude'}
            actual_dims = set(ds.dims)
            if actual_dims != expected_dims:
                print(f"  [X] FAULT: Invalid dimensional mapping. Expected {expected_dims}, found {actual_dims}")
                continue
            else:
                print("  [✓] Dimensionality is strictly 3D.")

            # 2. Chronological Continuity Check
            time_vector = ds.valid_time.values
            time_diffs = np.diff(time_vector)
            expected_diff = np.timedelta64(6, 'h')
            
            if not np.all(time_diffs == expected_diff):
                missing_indices = np.where(time_diffs != expected_diff)[0]
                print(f"  [X] FAULT: Temporal discontinuity detected at array index {missing_indices[0]}.")
                print(f"      Chronological jump from {time_vector[missing_indices[0]]} "
                      f"to {time_vector[missing_indices[0]+1]} (Expected +6h).")
                continue
            else:
                print("  [✓] Temporal axis is strictly continuous.")

            # 3. Real-Number Integrity Check
            print("  -> Computing boolean masks for NaN/Inf anomalies...")
            has_contamination = False
            faulty_vars = []
            
            # Iterate through variables to constrain RAM utilization
            for var in ds.data_vars:
                arr = ds[var].values
                if np.isnan(arr).any() or np.isinf(arr).any():
                    has_contamination = True
                    faulty_vars.append(var)
            
            if has_contamination:
                print(f"  [X] CRITICAL FAULT: Non-real values detected in physical variables: {faulty_vars}")
            else:
                print("  [✓] Tensor domain is strictly real (0 NaNs, 0 Infs).")
                
            ds.close()

        except Exception as e:
            print(f"  [X] EXECUTION FAULT: Failed to process {filename}. Traceback: {e}")

    print("\n==================================================")
    print("VERIFICATION PROTOCOL COMPLETE.")
    print("==================================================")

if __name__ == "__main__":
    verify_engineered_files()