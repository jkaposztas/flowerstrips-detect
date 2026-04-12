"""
Chips a VRT mosaic of Sentinel-2 super-resolved images using template chip boundaries.

This script processes a VRT file containing all Sentinel-2 super-resolved images by 
extracting chips at locations and sizes defined by template chips. Each output chip is 
resampled to a fixed 592x592 pixel size and saved as a GeoTIFF file. The script can be 
applied to both satellite imagery and rasterized flower strip label data.

Module Attributes:
    vrt_file (str): Path to the VRT mosaic file containing Sentinel-2 super-resolved images.
    CHIPS_DIR (str): Directory containing template chip GeoTIFF files used as cookie-cutters.
    OUT_DIR (str): Output directory where extracted chips will be saved.
    TARGET_SIZE (int): Target output chip dimensions (592x592 pixels).

Process:
    1. Opens the VRT file and reads its CRS, transform, band count, and data type.
    2. Iterates through all template chip files in CHIPS_DIR (sorted alphabetically).
    3. For each template chip:
       - Extracts its bounds and derives pixel resolution
       - Calculates a centered square region at TARGET_SIZE x TARGET_SIZE
       - Reprojects the VRT data to the target grid using bilinear resampling
       - Writes the extracted data as a GeoTIFF file with DEFLATE compression

"""
import os
from glob import glob
from multiprocessing import Pool

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, reproject

# to track progress: ls /workspaces/flowerstrips/data/flowerstripsdata/labels/output_cookie_cuts | wc -l
# set the path to the template chips, to the vrt, and to the output folder
# where the chips will be stored.
vrt_file = "/workspaces/flowerstrips/data/flowerstripsdata/labels/flowerstrips_repr.vrt"
CHIPS_DIR = "/workspaces/flowerstrips/data/template/bb_template_chips_3035"
OUT_DIR = "/workspaces/flowerstrips/data/flowerstripsdata/labels/output_cookie_cuts"
TARGET_SIZE = 592  # output must be 592 x 592. This is the optimal.
N_PROCESSES = 10    # adjust to your server

os.makedirs(OUT_DIR, exist_ok=True)

# list all the chips used as cookie-cutter
chip_files = sorted(glob(os.path.join(CHIPS_DIR, "*.tif")))  # limit to 100 for testing, remove [:100] for full run
def process_chip(chip_fp):
    out_name = os.path.splitext(os.path.basename(chip_fp))[0] + "_vrt_clip.tif"
    out_fp = os.path.join(OUT_DIR, out_name)
    
    if os.path.exists(out_fp):
        return out_name

    # open the vrt inside the function so multiprocessing works correctly
    with rasterio.open(vrt_file) as src:
        src_crs = src.crs
        src_transform = src.transform
        bands = src.count
        dtype = src.dtypes[0]
        src_nodata = src.nodata if src.nodata is not None else 0

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

            # allocate destination filled with nodata
            dst_arr = np.full((bands, TARGET_SIZE, TARGET_SIZE), src_nodata, dtype=dtype)

            # reproject VRT -> target grid
            for b in range(bands):
                reproject(
                    source=rasterio.band(src, b + 1),
                    destination=dst_arr[b],
                    src_transform=src_transform,
                    src_crs=src_crs,
                    dst_transform=dst_transform,
                    dst_crs=chip.crs,
                    resampling=Resampling.nearest, # change here to resampling=Resampling.nearest (instead of bilinear) for rasterized labels
                    dst_nodata=src_nodata,
                )

            # write output (preserve chip CRS and updated size/transform)
            out_profile = chip.profile.copy()
            out_profile.update({
                "height": TARGET_SIZE,
                "width": TARGET_SIZE,
                "transform": dst_transform,
                "crs": chip.crs,
                "count": bands,
                "dtype": dtype,
                "compress": "DEFLATE",
            })

            out_name = os.path.splitext(os.path.basename(chip_fp))[0] + "_vrt_clip.tif"
            out_fp = os.path.join(OUT_DIR, out_name)

            with rasterio.open(out_fp, "w", **out_profile) as dst:
                dst.write(dst_arr)

    return out_name


if __name__ == "__main__":
    with Pool(N_PROCESSES) as pool:
        pool.map(process_chip, chip_files)

    print("Done. Outputs in:", OUT_DIR)