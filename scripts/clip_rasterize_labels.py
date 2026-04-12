"""
Creates 592x592 flowerstrip label chips directly from polygon data using template chip boundaries.

This script processes a polygon layer with flowerstrips by clipping the polygons to the
locations defined by template chips and rasterizing them directly to the final chip grid.
Each output chip is saved as a GeoTIFF file and can later be paired with the satellite chips.
"""

import os
from glob import glob
from multiprocessing import Pool

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from shapely.geometry import box

# set the path to the polygon data, to the template chips, and to the output folder
POLYGON_FILE = "/workspaces/flowerstrips/data/flowerstripsdata/flowerstripsfinal/flowerstrips_overlayL3_089-011-parts090.gpkg"
CHIPS_DIR = "/workspaces/flowerstrips/data/template/bb_template_chips_3035"
OUT_DIR = "/workspaces/flowerstrips/data/flowerstripsdata/labels/output_cookie_cuts"
TARGET_SIZE = 592  # output must be 592 x 592
N_PROCESSES = 10   # adjust to your server

os.makedirs(OUT_DIR, exist_ok=True)

# list all the chips used as cookie-cutter
chip_files = sorted(glob(os.path.join(CHIPS_DIR, "*.tif")))  # limit to 200 for testing, remove [:200] for full run

# load polygon data once
gdf = gpd.read_file(POLYGON_FILE)
gdf = gdf.to_crs("EPSG:3035")  # make sure polygons are in the same CRS as the template chips


def process_chip(chip_fp):
    out_name = os.path.splitext(os.path.basename(chip_fp))[0] + "_vrt_clip.tif"
    out_fp = os.path.join(OUT_DIR, out_name)

    if os.path.exists(out_fp):
        return out_name

    with rasterio.open(chip_fp) as chip:
        # derive pixel size from chip and center the square on chip extent
        xmin, ymin, xmax, ymax = chip.bounds
        resx = abs(chip.transform.a)
        resy = abs(chip.transform.e)
        res = (resx + resy) / 2.0
        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0
        half = (TARGET_SIZE * res) / 2.0
        new_xmin, new_xmax = cx - half, cx + half
        new_ymin, new_ymax = cy - half, cy + half

        dst_transform = from_bounds(
            new_xmin, new_ymin, new_xmax, new_ymax, TARGET_SIZE, TARGET_SIZE
        )

        # clip polygon data to the target chip extent
        chip_bbox = box(new_xmin, new_ymin, new_xmax, new_ymax)
        gdf_chip = gdf[gdf.intersects(chip_bbox)].copy()

        if not gdf_chip.empty:
            gdf_chip = gdf_chip.clip(chip_bbox)

            # rasterize polygons directly to the final chip grid
            shapes = [(geom, 1) for geom in gdf_chip.geometry if geom is not None and not geom.is_empty]

            dst_arr = rasterize(
                shapes=shapes,
                out_shape=(TARGET_SIZE, TARGET_SIZE),
                transform=dst_transform,
                fill=0,
                dtype="uint8"
            )
        else:
            # empty chip if no flowerstrips intersect this extent
            dst_arr = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype="uint8")

        # write output
        out_profile = chip.profile.copy()
        out_profile.update({
            "height": TARGET_SIZE,
            "width": TARGET_SIZE,
            "transform": dst_transform,
            "crs": chip.crs,
            "count": 1,
            "dtype": "uint8",
            "compress": "DEFLATE",
            "nodata": 0,
        })

        with rasterio.open(out_fp, "w", **out_profile) as dst:
            dst.write(dst_arr, 1)

    return out_name


if __name__ == "__main__":
    with Pool(N_PROCESSES) as pool:
        pool.map(process_chip, chip_files)

    print("Done. Outputs in:", OUT_DIR)