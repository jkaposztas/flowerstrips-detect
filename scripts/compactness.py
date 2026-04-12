# -*- coding: utf-8 -*-
"""
Select linear polygons from INVEKOS code 090 using Polsby-Popper compactness.
Lower compactness = more linear / elongated features.
"""

import geopandas as gpd
from math import pi

# ----------------------------
# SETTINGS
# ----------------------------
input_gpkg = r"/workspaces/flowerstrips/data/flowerstripsdata/flowerstripsfinal/flowerstrips_overlayL3_090.gpkg"
output_gpkg = r"/workspaces/flowerstrips/data/flowerstripsdata/flowerstripsfinal/output_090_linear.gpkg"

code_column = "code"          # <- anpassen, falls deine Spalte anders heißt
target_code = "090"            # oder "090", je nach Datentyp in deiner Tabelle
compactness_threshold = 0.25   # alles darunter = eher linear
target_crs = "EPSG:3035"

# ----------------------------
# FUNCTIONS
# ----------------------------
def pp_compactness(geom):
    """Polsby-Popper compactness index."""
    if geom is None or geom.is_empty:
        return None

    p = geom.length
    a = geom.area

    if p == 0:
        return None

    return (4 * pi * a) / (p * p)

# ----------------------------
# LOAD DATA
# ----------------------------
gdf = gpd.read_file(input_gpkg)

print(f"Original feature count: {len(gdf)}")
print(f"Original CRS: {gdf.crs}")

# Reproject if needed
if gdf.crs != target_crs:
    gdf = gdf.to_crs(target_crs)
    print(f"Reprojected to: {target_crs}")

# ----------------------------
# SELECT CODE 090
# ----------------------------
gdf_090 = gdf[gdf[code_column] == target_code].copy()

print(f"Code 090 feature count: {len(gdf_090)}")

# ----------------------------
# CALCULATE COMPACTNESS
# ----------------------------
gdf_090["compactness"] = gdf_090.geometry.apply(pp_compactness)
gdf_090["area_m2"] = gdf_090.geometry.area

# ----------------------------
# FILTER LINEAR FEATURES
# ----------------------------
gdf_090_linear = gdf_090[gdf_090["compactness"] < compactness_threshold].copy()

print(f"Linear 090 features (< {compactness_threshold}): {len(gdf_090_linear)}")
print(f"Total area before filtering: {gdf_090['area_m2'].sum() / 10000:.2f} ha")
print(f"Total area after filtering: {gdf_090_linear['area_m2'].sum() / 10000:.2f} ha")

# ----------------------------
# SAVE OUTPUT
# ----------------------------
gdf_090_linear.to_file(output_gpkg, driver="GPKG")

print("Done.")
print(f"Saved to: {output_gpkg}")