import xarray as xr
import numpy as np

# Load the local NetCDF file
ds = xr.open_dataset("./data/processed/engineered_1996.nc")

# Extract the float32 array
lats = ds['latitude'].values

# Serialize strictly as a NumPy binary file into the transfer directory
np.save("./data/processed/tensors/latitudes.npy", lats)

print(f"Serialized {len(lats)} latitude coordinates successfully.")