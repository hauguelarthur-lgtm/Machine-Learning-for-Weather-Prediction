import cdsapi
import os

# Initialize the CDS client
c = cdsapi.Client()

YEARS = [str(year) for year in range(1995, 1996)]
MONTHS = [f"{month:02d}" for month in range(1, 13)]
DAYS = [f"{day:02d}" for day in range(1, 32)]
TIMES = ['00:00', '06:00', '12:00', '18:00']
AREA_FRANCE = [54.50, -6.25, 38.75, 9.50]
GRID_RESOLUTION = [0.25, 0.25]

OUTPUT_DIR = "./era5_france_raw"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def download_surface_data(year, month):
    """Extracts 2D surface boundary conditions per month."""
    filename = os.path.join(OUTPUT_DIR, f"surface_{year}_{month}.nc")
    if os.path.exists(filename):
        print(f"Skipping {filename}")
        return

    print(f"Requesting Surface Data: {year}-{month}")
    try:
        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'variable': [
                    '2m_temperature', 'surface_pressure', 
                    '10m_u_component_of_wind', '10m_v_component_of_wind', 
                    'total_precipitation',
                    'land_sea_mask', 'geopotential', 'soil_type',
                    'convective_available_potential_energy',
                    'surface_net_solar_radiation', 'top_net_thermal_radiation',
                    'volumetric_soil_water_layer_1', 'boundary_layer_height'
                ],
                'year': year,
                'month': month,
                'day': DAYS,
                'time': TIMES,
                'area': AREA_FRANCE,
                'grid': GRID_RESOLUTION,
            },
            filename
        )
    except Exception as e:
        print(f"Failed Surface {year}-{month}: {e}")

def download_pressure_data(year, month):
    """Extracts 3D kinematic and thermodynamic profiles per month."""
    filename = os.path.join(OUTPUT_DIR, f"pressure_{year}_{month}.nc")
    if os.path.exists(filename):
        print(f"Skipping {filename}")
        return

    print(f"Requesting Pressure Data: {year}-{month}")
    try:
        c.retrieve(
            'reanalysis-era5-pressure-levels',
            {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'variable': [
                    'geopotential', 'specific_humidity', 'temperature',
                    'u_component_of_wind', 'v_component_of_wind',
                    'vertical_velocity', 
                    'specific_cloud_liquid_water_content', 'specific_cloud_ice_water_content',
                    'divergence', 'fraction_of_cloud_cover'
                ],
                'pressure_level': ['250', '500', '850', '1000'],
                'year': year,
                'month': month,
                'day': DAYS,
                'time': TIMES,
                'area': AREA_FRANCE,
                'grid': GRID_RESOLUTION,
            },
            filename
        )
    except Exception as e:
        print(f"Failed Pressure {year}-{month}: {e}")

# Execution Loop
for year in YEARS:
    for month in MONTHS:
        download_surface_data(year, month)
        download_pressure_data(year, month)

print("Data extraction complete.")