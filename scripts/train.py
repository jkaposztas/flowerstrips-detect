#!/usr/bin/env python3
# ============================================================
# Flowerstrips U-Net Training — standalone script
#
# Start (one band configuration per call):
#   CUDA_VISIBLE_DEVICES=3 nohup python -u train.py > grid_rgb_nir.log 2>&1 &
#
# Change the band configuration ONLY at the bottom of the CONFIG section, then restart.
# The 9 hyperparameter combinations (weight_decay x batch_size) run automatically
# one after the other. Completed runs are skipped via the DONE marker
# -> if it crashes, simply restart; only the missing runs will be executed.
# ============================================================

import os
import csv
import random
from glob import glob
from datetime import datetime

import numpy as np
import pandas as pd
import rasterio
import tifffile

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import ParameterGrid
import segmentation_models_pytorch as smp

# ============================================================
# Adjust Band-CONFIG
# ============================================================
#BAND_CONFIG = "rgb_nir"          # <-- Change Band Combination
BAND_CONFIG = "rgb_nir_swir"

OPATH         = "..."
LABELS_FOLDER = os.path.join(OPATH, "Labels_filtered")
MODELFOLDER   = os.path.join(OPATH, "model")
ALL_RUNS_LOG  = os.path.join(MODELFOLDER, "all_runs.csv")
TEST_CHIPS_CSV = os.path.join(OPATH, "output/test_chips_list.csv")
TRAIN_CHIPS_CSV = os.path.join(OPATH, "output/train_chips_list.csv")
VAL_CHIPS_CSV   = os.path.join(OPATH, "output/val_chips_list.csv")

CHIP_SIZE = 592
EPOCHS    = 200
PATIENCE  = 10            # Early stopping
POS_WEIGHT = 15.0
LR = 1e-3
VAL_FRACTION = 0.2
NUM_WORKERS = 0           # deliberately 0 for reproducibility
SEED = 1234

if BAND_CONFIG == "rgb_nir":
    IMAGES_FOLDER = os.path.join(OPATH, "SR-Chips/RG_NIR")
    MONTHBANDS = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    NORM_COEF = [2806, 3217, 7020,
                 2814, 3219, 6248,
                 2744, 3186, 6296]
elif BAND_CONFIG == "rgb_nir_swir":
    IMAGES_FOLDER = os.path.join(OPATH, "SR-Chips/RG_NIR_SWIR")
    MONTHBANDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    NORM_COEF = [2806, 3217, 7020, 5049, 4916,
                 2814, 3219, 6248, 5280, 4884,
                 2744, 3186, 6296, 5336, 4705]
else:
    raise ValueError(f"Unknown BAND_CONFIG: {BAND_CONFIG}")

NUM_CHANNELS = len(MONTHBANDS)

param_grid = {
    'weight_decay': [0, 1e-4, 1e-3],
    'batch_size':   [4, 8, 16],
}
PARAM_COMBINATIONS = list(ParameterGrid(param_grid))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Reproducibility
# ============================================================
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Dataset
# ============================================================
class FlowerstripsDataset(Dataset):
    def __init__(self, image_paths, label_paths, monthbands, norm_coef, chip_size, augment=False):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.monthbands = monthbands
        self.norm_coef = norm_coef
        self.augment = augment
        self.chip_size = chip_size
        assert len(self.image_paths) == len(self.label_paths)

    def __len__(self): # Returns the length of the data set
        return len(self.image_paths)

    def __getitem__(self, idx): # Exactly 1 chip is supplied here; idx = the chip that is now being used for training
        img = tifffile.imread(self.image_paths[idx])   # (H, W, C)
        lbl = tifffile.imread(self.label_paths[idx])   # (H, W)

        img = img[..., self.monthbands]
        h, w, c = img.shape
        if h < self.chip_size or w < self.chip_size:
            pad_h = max(0, self.chip_size - h)
            pad_w = max(0, self.chip_size - w)
            img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant", constant_values=0)
        img = img[:self.chip_size, :self.chip_size, :]

        img = img.astype(np.float32) # For BCEWithLogitsLoss, the target must be a float in the range {0.0, 1.0} — check that your FlowerstripsDataset is cast correctly (.float() / astype(np.float32)).
        n_bands = img.shape[-1]
        if n_bands == len(self.norm_coef):
            for i in range(n_bands):
                img[..., i] /= self.norm_coef[i]

        img = np.transpose(img, (2, 0, 1))   # (C, H, W)

        if lbl.ndim != 2:
            raise ValueError(f"Unexpected label shape: {lbl.shape}")
        lh, lw = lbl.shape
        if lh < self.chip_size or lw < self.chip_size:
            pad_h = max(0, self.chip_size - lh)
            pad_w = max(0, self.chip_size - lw)
            lbl = np.pad(lbl, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
        lbl = lbl[:self.chip_size, :self.chip_size]
        lbl = lbl.astype(np.float32)[None, ...]   # (1, H, W)

        if self.augment:
            if random.random() > 0.5:
                img = np.flip(img, axis=2).copy()
                lbl = np.flip(lbl, axis=2).copy()
            if random.random() > 0.5:
                img = np.flip(img, axis=1).copy()
                lbl = np.flip(lbl, axis=1).copy()
        return torch.from_numpy(img), torch.from_numpy(lbl)


# ============================================================
# Stratified split
# The test CSV is read ONLY; it is never regenerated.
# The split is seed-dependent -> create it globally once; it will be identical for all runs.
# ============================================================
def has_positive(label_path):
    with rasterio.open(label_path) as src:
        arr = src.read(1)
    return arr.sum() > 0


# ============================================================
# Split — now loaded from pre-prepared CSV files
# (spatially leakage-free block split, see build_block_split_with_cleanup.py)
# ============================================================
def build_split():
    train_names = pd.read_csv(TRAIN_CHIPS_CSV)["0"].to_list()
    val_names   = pd.read_csv(VAL_CHIPS_CSV)["0"].to_list()
    test_names  = pd.read_csv(TEST_CHIPS_CSV)["0"].to_list()

    train_images = [os.path.join(IMAGES_FOLDER, n) for n in train_names]
    train_labels = [os.path.join(LABELS_FOLDER, n) for n in train_names]
    val_images   = [os.path.join(IMAGES_FOLDER, n) for n in val_names]
    val_labels   = [os.path.join(LABELS_FOLDER, n) for n in val_names]

    print(f"Train: {len(train_images)} | Val: {len(val_images)} | "
          f"Test: {len(test_names)}", flush=True)

    return train_images, train_labels, val_images, val_labels


# ============================================================
# Metrics: segmentation_models_pytorch (smp)
# ============================================================
def batch_stats(outputs, labels, threshold=0.3):
    """Rohe per-Chip-Zaehler (tp, fp, fn, tn) via smp.get_stats.
    Erwartet ROHE logits; Sigmoid+Threshold intern. Rueckgabe: LongTensor je
    Shape (N, 1), damit sie ueber Batches konkateniert und dann mit den
    smp-Reduktionen (micro / micro-imagewise) ausgewertet werden koennen."""
    prob   = torch.sigmoid(outputs)
    target = (labels > 0.5).long()
    tp, fp, fn, tn = smp.metrics.get_stats(prob, target, mode="binary",
                                           threshold=threshold)
    return tp, fp, fn, tn


def iou_pytorch(outputs, labels, threshold=0.3):
    """Per-Chip-IoU, ueber Chips gemittelt (micro-imagewise), leere Chips=1.0.
    Fuer train_iou pro Batch verwendet (ein Skalar je Batch)."""
    tp, fp, fn, tn = batch_stats(outputs, labels, threshold)
    return float(smp.metrics.iou_score(tp, fp, fn, tn,
                 reduction="micro-imagewise", zero_division=1.0))


# ============================================================
# A training run:
# ============================================================
def run_experiment(params, split):
    weight_decay = params["weight_decay"]
    batch_size   = params["batch_size"]

    model_name = (f"FS_unet_resnet34_pw{POS_WEIGHT:.0f}_bs{batch_size}"
                  f"_wd{weight_decay:.0e}_{BAND_CONFIG}")
    model_path = os.path.join(MODELFOLDER, f"{model_name}.pt")
    log_path   = os.path.join(MODELFOLDER, f"{model_name}.log")
    done_path  = os.path.join(MODELFOLDER, f"{model_name}.DONE")

    if os.path.exists(done_path):
        print(f"[skip] {model_name} (DONE existiert)", flush=True)
        return

    print(f"\n{'='*60}\n[start] {model_name}\n{'='*60}", flush=True)

    # --- Reproducibility per fresh run ---
    set_seed(SEED)

    train_images, train_labels, val_images, val_labels = split
    train_ds = FlowerstripsDataset(train_images, train_labels, MONTHBANDS, NORM_COEF, CHIP_SIZE, augment=True)
    val_ds   = FlowerstripsDataset(val_images,   val_labels,   MONTHBANDS, NORM_COEF, CHIP_SIZE, augment=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True) # drop_last=False ist aktiv → alle Chips werden garantiert jede Epoche gesehen, unabhängig davon ob N durch 4 teilbar ist. shuffle=True ohne sampler=-Argument bedeutet: PyTorch nutzt intern einen RandomSampler
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # --- Model / Optimizer / Scaler PRO RUN (new) ---
    model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet",
                     in_channels=NUM_CHANNELS, classes=1).to(device)

    pos_weight = torch.tensor([POS_WEIGHT], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    scaler = GradScaler()

    # --- .log for this run ---
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "train_iou",
                                "val_iou", "val_precision", "val_recall"])

    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_row = None

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        train_iou = 0.0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels) 
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * images.size(0)
            train_iou += iou_pytorch(outputs.detach().cpu(), labels.cpu()) * images.size(0)

        train_loss /= len(train_loader.dataset)
        train_iou  /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        val_tp, val_fp, val_fn, val_tn = [], [], [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)                
                tp, fp, fn, tn = batch_stats(outputs.cpu(), labels.cpu(),
                                             threshold=0.3)
                val_tp.append(tp); val_fp.append(fp)
                val_fn.append(fn); val_tn.append(tn)

        val_loss /= len(val_loader.dataset)

        # Concatenate the counters for all Val-chips -> (N_val, 1)
        VTP = torch.cat(val_tp); VFP = torch.cat(val_fp)
        VFN = torch.cat(val_fn); VTN = torch.cat(val_tn)

        # val_iou: per-chip IoU, averaged across chips (empty chips = 1.0)
        val_iou = float(smp.metrics.iou_score(
            VTP, VFP, VFN, VTN, reduction="micro-imagewise", zero_division=1.0))
        # val_precision / val_recall: pooled across all Val pixels (micro)
        val_precision = float(smp.metrics.precision(
            VTP, VFP, VFN, VTN, reduction="micro"))
        val_recall = float(smp.metrics.recall(
            VTP, VFP, VFN, VTN, reduction="micro"))

        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} "
              f"- Train IoU: {train_iou:.4f} - Val IoU: {val_iou:.4f} "
              f"- Val Prec: {val_precision:.4f} - Val Rec: {val_recall:.4f} ")

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch+1, train_loss, val_loss, train_iou,
                                    val_iou, val_precision, val_recall])

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            epochs_no_improve = 0
            best_row = [model_name, datetime.now().strftime("%Y-%m-%d"),
                        epoch+1, train_loss, val_loss, train_iou, val_iou,
                        val_precision, val_recall]
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print("Early stopping triggered.", flush=True)
                break

    # Batch file (one line per run)
    if best_row is not None:
        with open(ALL_RUNS_LOG, "a", newline="") as f:
            csv.writer(f).writerow(best_row)

    # Resume-Marker
    open(done_path, "w").close()
    print(f"[done] {model_name} - best Val Loss: {best_val_loss:.4f}", flush=True)


# ============================================================
# Main
# ============================================================
def main():
    os.makedirs(MODELFOLDER, exist_ok=True)
    print(f"Device: {device} | Band-Config: {BAND_CONFIG} | "
          f"{len(PARAM_COMBINATIONS)} combinations", flush=True)

    if not os.path.exists(ALL_RUNS_LOG):
        with open(ALL_RUNS_LOG, "w", newline="") as f:
            csv.writer(f).writerow(["model_name", "date", "epoch", "train_loss",
                                    "val_loss", "train_iou", "val_iou",
                                    "val_precision", "val_recall"])

    split = build_split()

    for i, params in enumerate(PARAM_COMBINATIONS):
        print(f"\n### Grid {i+1}/{len(PARAM_COMBINATIONS)}: {params}", flush=True)
        run_experiment(params, split)

    print("\nAll runs for this band configuration have been completed.", flush=True)


if __name__ == "__main__":
    main()