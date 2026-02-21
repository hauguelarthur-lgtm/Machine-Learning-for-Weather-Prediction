import os
import time
from pathlib import Path
import cdsapi

# 1) Connexion à l'API CDS
c = cdsapi.Client()

# 2) Définition de la zone
AREA_CA = [84.0, -142.0, 41.0, -52.0]
# CHANGE 1: Use 0.1 for ERA5-Land (Native resolution) instead of 0.25
GRID = [0.1, 0.1] 

# 3) Période à télécharger
START_YEAR = 2001
END_YEAR = 2025
YEARS = list(range(START_YEAR, END_YEAR + 1))
MONTHS = list(range(1, 13))
# Helper for all days in a month (1-31)
DAYS = [f"{d:02d}" for d in range(1, 32)]
TIMES = [f"{h:02d}:00" for h in range(24)]

# 4) Dossier de sortie
OUT_DIR = Path("era5_land_ca_t2m") # Renamed folder for clarity
OUT_DIR.mkdir(parents=True, exist_ok=True)

def retrieve_one_month(year: int, month: int, out_path: Path) -> None:
    """Download one month of ERA5-LAND t2m."""
    
    request = {
        # CHANGE 2: REMOVED "product_type": "reanalysis" (Causes error for Land)
        "variable": ["2m_temperature"],
        "year": f"{year:04d}",
        "month": f"{month:02d}",
        "day": DAYS,
        "time": TIMES,
        "area": AREA_CA,
        "format": "netcdf", 
        # Optional: You can remove "grid" to get default, 
        # or keep it to force interpolation.
        # "grid": GRID, 
    }

    start = time.time()
    
    # CHANGE 3: Dataset name changed to 'reanalysis-era5-land'
    c.retrieve(
        "reanalysis-era5-land",
        request,
        str(out_path),
    )
    
    elapsed = time.time() - start
    print(f"OK {year}-{month:02d} -> {out_path.name} ({elapsed:.1f}s)")

def main() -> None:
    total_start = time.time()
    downloaded = 0
    skipped = 0

    print(f"Starting download for ERA5-Land ({START_YEAR}-{END_YEAR})...")

    for year in YEARS:
        for month in MONTHS:
            # Updated filename to reflect "land"
            out_file = OUT_DIR / f"era5_land_t2m_ca_{year:04d}_{month:02d}.nc"

            if out_file.exists() and out_file.stat().st_size > 0:
                skipped += 1
                continue

            for attempt in range(1, 4):
                try:
                    retrieve_one_month(year, month, out_file)
                    downloaded += 1
                    break
                except Exception as e:
                    print(f"WARN {year}-{month:02d} attempt {attempt} failed: {e}")
                    if attempt == 3:
                        # Optional: Don't crash the whole script for one month
                        print(f"ERROR: Could not download {year}-{month:02d}. Moving on.")
                    time.sleep(5)

    total_elapsed = time.time() - total_start
    print("\nTerminé !")
    print(f"Dossier : {OUT_DIR.resolve()}")
    print(f"Téléchargés : {downloaded}")
    print(f"Déjà présents : {skipped}")
    print(f"Temps total : {total_elapsed/60:.1f} min")

if __name__ == "__main__":
    main()