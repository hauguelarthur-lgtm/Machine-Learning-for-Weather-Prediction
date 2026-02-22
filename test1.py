import xarray as xr
import numpy as np
import pandas as pd

datatest = xr.open_dataset("era5_france_raw/pressure_1996_01.nc")
print(size(datatest))