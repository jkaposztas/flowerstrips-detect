# -*- coding: utf-8 -*-

from pathlib import Path
import rasterio
import numpy as np
import shutil

# ----------------------------
# SETTINGS
# ----------------------------
LABEL_DIR = Path(r"/workspaces/flowerstrips/data/flowerstripsdata/labels/output_cookie_cuts")
OUT_DIR = Path(r"/workspaces/flowerstrips/data/flowerstripsdata/labels/selected")

OUT_DIR.mkdir(exist_ok=True)

MAX_CHIPS = 100 

# ----------------------------
# FUNCTION
# ----------------------------
def has_label(raster_path):
    with rasterio.open(raster_path) as src:
        data = src.read(1)

        return np.any(data != 0)

# ----------------------------
# MAIN
# ----------------------------
count = 0

for tif in sorted(LABEL_DIR.glob("*.tif")):
    if has_label(tif):
        shutil.copy2(tif, OUT_DIR / tif.name)
        count += 1
        print(f"Copied: {tif.name}")

    if count >= MAX_CHIPS:
        break

print("Done.")