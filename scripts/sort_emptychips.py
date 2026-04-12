# -*- coding: utf-8 -*-
"""
Keep only Sentinel-2 SR chips that
1) exist in all required month folders
2) are not empty

The script copies only valid chips into a new output directory,
keeping one subfolder per month.
"""

from pathlib import Path
import shutil
import rasterio
import numpy as np

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

MONTH_DIRS = {
    "april": Path(r"/workspaces/flowerstrips/data/satellite/output_cookie_cuts/april"),
    "june": Path(r"/workspaces/flowerstrips/data/satellite/output_cookie_cuts/june"),
    "august": Path(r"/workspaces/flowerstrips/data/satellite/output_cookie_cuts/august"),
    # "december": Path(r"/workspaces/flowerstrips/data/satellite/output_cookie_cuts/december"),
}

OUT_BASE = Path(r"/workspaces/flowerstrips/data/satellitechips_filtered")

FILE_EXTENSION = "*.tif"

# --------------------------------------------------
# FUNCTIONS
# --------------------------------------------------

def get_chip_id(filepath: Path) -> str:
    return filepath.stem


def is_empty_chip(raster_path):
    """
    Returns True if the chip is unusable.
    A chip is considered unusable if ANY band is:
    - fully masked / nodata
    - all NaN
    - all zero
    - has no finite valid values
    """
    with rasterio.open(raster_path) as src:
        data = src.read(masked=True)

        for b in range(data.shape[0]):
            band = data[b]

            # all masked
            if np.ma.count(band) == 0:
                return True

            # valid unmasked values
            valid = band.compressed()

            if valid.size == 0:
                return True

            # keep only finite values
            valid = valid[np.isfinite(valid)]

            if valid.size == 0:
                return True

            # all zero
            if np.all(valid == 0):
                return True

        return False


def collect_files(folder: Path) -> dict:
    files = sorted(folder.glob(FILE_EXTENSION))
    return {get_chip_id(fp): fp for fp in files}


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():
    print("Collecting chip files...")

    month_files = {}
    for month, folder in MONTH_DIRS.items():
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder}")
        month_files[month] = collect_files(folder)
        print(f"{month}: {len(month_files[month])} files found")

    common_ids = set.intersection(*(set(d.keys()) for d in month_files.values()))
    print(f"\nCommon chip IDs across all months: {len(common_ids)}")

    valid_ids = []
    invalid_missing_or_empty = []

    print("\nChecking whether common chips are empty...")
    for chip_id in sorted(common_ids):
        keep_chip = True

        for month, files_dict in month_files.items():
            chip_path = files_dict[chip_id]

            if is_empty_chip(chip_path):
                keep_chip = False
                invalid_missing_or_empty.append((chip_id, month, chip_path.name))
                print(f"Removed: {chip_id} | month: {month} | file: {chip_path.name}")
                break

        if keep_chip:
            valid_ids.append(chip_id)

    print(f"\nInvalid chip IDs removed: {len(invalid_missing_or_empty)}")
    print(f"Valid chip IDs present and non-empty in all months: {len(valid_ids)}")

    for month in MONTH_DIRS.keys():
        (OUT_BASE / month).mkdir(parents=True, exist_ok=True)

    print("\nCopying valid chips...")
    for chip_id in valid_ids:
        for month, files_dict in month_files.items():
            src = files_dict[chip_id]
            dst = OUT_BASE / month / src.name
            shutil.copy2(src, dst)

    print("\nDone.")
    print(f"Filtered chips saved to: {OUT_BASE}")

    valid_ids_txt = OUT_BASE / "valid_chip_ids.txt"
    with open(valid_ids_txt, "w", encoding="utf-8") as f:
        for chip_id in valid_ids:
            f.write(chip_id + "\n")

    print(f"Valid chip ID list saved to: {valid_ids_txt}")


if __name__ == "__main__":
    main()