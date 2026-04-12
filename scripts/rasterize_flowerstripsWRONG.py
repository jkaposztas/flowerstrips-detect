import geopandas as gpd
"""
Rasterize flower strip polygons to match a reference satellite imagery grid.

This script converts polygon geometries from a GeoPackage file into a raster format,
aligned with a reference satellite imagery grid. The output raster has a pixel value of 1
for areas covered by flower strips and 0 for other areas.

Workflow:
    1. Load polygon layer from GeoPackage (flowerstrips_BB_L3.gpkg)
    2. Reproject to EPSG:3035 if necessary
    3. Load reference raster to obtain transformation parameters, dimensions, and CRS
    4. Rasterize all polygons with value 1 on a background of 0
    5. Write output to a GeoTIFF file

Output:
    Location: /workspaces/flowerstrips/data/flowerstripsdata/labels/
    Filename: flowerstrips_3035.tif
    Format: GeoTIFF with DEFLATE compression
    Data type: uint8 (unsigned 8-bit integer)
    Bands: 1 (single band raster)
"""
import rasterio
import numpy as np
from rasterio.features import rasterize

# Polygonlayer laden
gdf = gpd.read_file("/workspaces/flowerstrips/data/flowerstripsdata/flowerstripsfinal/flowerstrips_BB_L3.gpkg")

# CRS prüfen / umprojizieren
# gdf.head() # Check the first few rows of the GeoDataFrame
print(gdf.crs)
if gdf.crs != "EPSG:3035":
    gdf = gdf.to_crs("EPSG:3035")

# Referenzraster laden, um die Transformationsparameter zu erhalten
with rasterio.open("/workspaces/flowerstrips/data/satellite/satellite_data3035/april/satellite_april_repr.vrt") as src:
    transform = src.transform
    width = src.width
    height = src.height 
    crs = src.crs

# Alle Polygone als Wert 1 rasterisieren
shapes = [(geom, 1) for geom in gdf.geometry]

raster = rasterize(
    shapes=shapes,
    out_shape=(height, width),
    transform=transform,
    fill=0,
    dtype="uint8"
)

# Output speichern
with rasterio.open(
    "/workspaces/flowerstrips/data/flowerstripsdata/labels/flowerstrips_3035.tif", # ändern zu NRW!!!    "w",
    driver="GTiff",
    height=height,
    width=width,
    count=1,
    dtype="uint8",
    crs=crs,
    transform=transform,
    compress="DEFLATE"
) as dst:
    dst.write(raster, 1)

print("Raster fertig.")