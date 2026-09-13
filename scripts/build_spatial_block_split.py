#!/usr/bin/env python3
# ============================================================
# Two-stage spatial split for a DENSE, near-continuous chip grid
#
# Stage 1: coarse grid block assignment (train/val/test), for
#          the overall size/positive-class balance.
# Stage 2: buffer cleanup - any individual train/val chip whose
#          footprint actually overlaps a test chip is DROPPED
#          from train/val (not reassigned, not the whole block -
#          just that one chip). Same for train chips overlapping
#          val chips. This guarantees zero leakage at the cost of
#          losing a modest number of boundary chips.
# ============================================================

import os
import random
from glob import glob
from collections import defaultdict

import rasterio
import pandas as pd
import geopandas as gpd
from shapely.geometry import box

# ============================================================
# CONFIG
# ============================================================
LABELS_FOLDER = ".../BB/Labels_filtered"
OUT_DIR       = ".../BB"

# Larger than before - with a dense/continuous grid, a bigger block
# means a smaller fraction of chips sit near a block boundary, so
# fewer chips get dropped in stage 2.
BLOCK_SIZE_M = 10000

TARGET_FRACS = {"train": 0.64, "val": 0.16, "test": 0.20}
SEED = 1234


def get_chip_info(fp):
    with rasterio.open(fp) as src:
        xmin, ymin, xmax, ymax = src.bounds
        cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
        has_pos = bool(src.read(1).sum() > 0)
    return cx, cy, has_pos, box(xmin, ymin, xmax, ymax)


def assign_block(cx, cy, block_size):
    return (int(cx // block_size), int(cy // block_size))


# ============================================================
# Stage 1: grid block assignment (same logic as before)
# ============================================================
def build_blocks(labels_folder, block_size):
    label_files = sorted(glob(os.path.join(labels_folder, "*.tif")))
    print(f"Found {len(label_files)} label chips")

    blocks = defaultdict(lambda: {"names": [], "n_total": 0, "n_pos": 0})
    chip_bboxes = {}

    for i, fp in enumerate(label_files):
        cx, cy, has_pos, bbox = get_chip_info(fp)
        name = os.path.basename(fp)
        block_id = assign_block(cx, cy, block_size)

        blocks[block_id]["names"].append(name)
        blocks[block_id]["n_total"] += 1
        blocks[block_id]["n_pos"] += int(has_pos)
        chip_bboxes[name] = bbox

        if (i + 1) % 2000 == 0:
            print(f"  processed {i + 1} / {len(label_files)}")

    print(f"Chips grouped into {len(blocks)} blocks (block size {block_size}m)")
    return blocks, chip_bboxes


def greedy_balanced_assignment(blocks, target_fracs, seed=SEED):
    random.seed(seed)
    block_list = list(blocks.items())
    random.shuffle(block_list)
    block_list.sort(key=lambda kv: kv[1]["n_total"], reverse=True)

    grand_total = sum(b["n_total"] for _, b in block_list)
    grand_pos = sum(b["n_pos"] for _, b in block_list)

    current = {g: {"total": 0, "pos": 0} for g in target_fracs}
    assignment = {g: [] for g in target_fracs}

    for block_id, b in block_list:
        scores = {}
        for g, frac in target_fracs.items():
            target_total = frac * grand_total
            target_pos = frac * grand_pos
            deficit_total = (target_total - current[g]["total"]) / max(target_total, 1)
            deficit_pos = (target_pos - current[g]["pos"]) / max(target_pos, 1)
            scores[g] = deficit_total + deficit_pos
        best_group = max(scores, key=scores.get)
        assignment[best_group].extend(b["names"])
        current[best_group]["total"] += b["n_total"]
        current[best_group]["pos"] += b["n_pos"]

    print("\nStage 1 (before cleanup):")
    for g in target_fracs:
        pct = 100 * current[g]["total"] / grand_total
        print(f"  {g:5s}: {current[g]['total']:5d} chips ({pct:.1f}%)")

    return assignment


# ============================================================
# Stage 2: drop individual chips causing cross-split leakage.
# Priority: keep test fully intact (most important for unbiased
# final evaluation) -> drop leaking chips from train/val instead.
# Then also clean train vs val (drop from train, since it's larger).
# ============================================================
def drop_leaking_chips(assignment, chip_bboxes):
    def to_gdf(names):
        return gpd.GeoDataFrame({"name": names}, geometry=[chip_bboxes[n] for n in names])

    test_gdf = to_gdf(assignment["test"])
    train_gdf = to_gdf(assignment["train"])
    val_gdf = to_gdf(assignment["val"])

    # train/val chips overlapping test -> drop from train/val
    train_val_gdf = pd.concat([train_gdf, val_gdf], ignore_index=True)
    overlap_test = gpd.sjoin(train_val_gdf, test_gdf, predicate="intersects")
    drop_vs_test = set(overlap_test["name_left"])

    train_names = [n for n in assignment["train"] if n not in drop_vs_test]
    val_names = [n for n in assignment["val"] if n not in drop_vs_test]

    print(f"\nDropped {len(drop_vs_test)} train/val chips overlapping test")

    # remaining train chips overlapping val -> drop from train
    train_gdf2 = to_gdf(train_names)
    val_gdf2 = to_gdf(val_names)
    overlap_val = gpd.sjoin(train_gdf2, val_gdf2, predicate="intersects")
    drop_vs_val = set(overlap_val["name_left"])

    train_names_final = [n for n in train_names if n not in drop_vs_val]
    print(f"Dropped {len(drop_vs_val)} additional train chips overlapping val")

    return train_names_final, val_names, assignment["test"]


if __name__ == "__main__":
    blocks, chip_bboxes = build_blocks(LABELS_FOLDER, BLOCK_SIZE_M)
    assignment = greedy_balanced_assignment(blocks, TARGET_FRACS)
    train_names, val_names, test_names = drop_leaking_chips(assignment, chip_bboxes)

    print("\nFinal split (after cleanup):")
    total = len(train_names) + len(val_names) + len(test_names)
    for label, names in [("train", train_names), ("val", val_names), ("test", test_names)]:
        print(f"  {label:5s}: {len(names):5d} chips ({100 * len(names) / total:.1f}%)")

    for label, names in [("train", train_names), ("val", val_names), ("test", test_names)]:
        out_fp = os.path.join(OUT_DIR, f"{label}_chips_list.csv")
        pd.DataFrame({"0": names}).to_csv(out_fp, index=False)
        print(f"Saved {len(names)} chip names to {out_fp}")