import os
import json
import numpy as np
import geopandas as gpd

current_dir = os.path.dirname(os.path.abspath(__file__))
shapefile_path = os.path.abspath(os.path.join(current_dir, '..', 'data', 'shapefiles', 'departement.shp'))
out_dir = os.path.abspath(os.path.join(current_dir, '..', 'data', 'processed', 'tensors'))

# Ensure bounding box matches your ERA5 physical spatial domain strictly
H, W = 64, 64 # Replace with actual spatial dimensions (H, W) from P_initial
lat_array = np.linspace(54.50, 38.75, H)
lon_array = np.linspace(-6.25, 9.50, W)

gdf = gpd.read_file(shapefile_path).to_crs(epsg=4326)

shp_lons, shp_lats = [], []
for geom in gdf.geometry:
    if geom is None: continue
    if geom.type == 'Polygon':
        x, y = geom.exterior.coords.xy
        shp_lons.extend(x.tolist() + [None])
        shp_lats.extend(y.tolist() + [None])
    elif geom.type == 'MultiPolygon':
        for poly in geom.geoms:
            x, y = poly.exterior.coords.xy
            shp_lons.extend(x.tolist() + [None])
            shp_lats.extend(y.tolist() + [None])

france_boundary = gdf.geometry.unary_union
Lons, Lats = np.meshgrid(lon_array, lat_array)
points_geoseries = gpd.GeoSeries(gpd.points_from_xy(Lons.flatten(), Lats.flatten()))
mask_matrix = points_geoseries.within(france_boundary).values.reshape(Lons.shape)

np.save(os.path.join(out_dir, "france_mask.npy"), mask_matrix)
with open(os.path.join(out_dir, "france_boundaries.json"), "w") as f:
    json.dump({"lons": shp_lons, "lats": shp_lats}, f)

print(f"Topological arrays serialized to {out_dir}")