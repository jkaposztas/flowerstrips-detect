#!/usr/bin/env python3
# ============================================================
# Flowerstrips U-Net Training — standalone script
#
# Start (eine Band-Config je Aufruf):
#   CUDA_VISIBLE_DEVICES=3 nohup python -u train.py > grid_rgb_nir.log 2>&1 &
#
# Band-Config NUR unten in der CONFIG-Sektion umstellen, dann erneut starten.
# Die 9 Hyperparameter-Kombis (weight_decay x batch_size) laufen automatisch
# nacheinander durch. Abgeschlossene Runs werden per DONE-Marker übersprungen
# -> bei Absturz einfach neu starten, es laufen nur die fehlenden Runs.
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
# CONFIG  — HIER die Band-Config je Lauf umstellen
# ============================================================
BAND_CONFIG = "rgb_nir"          # <-- Change Band Combination
#BAND_CONFIG = "rgb_nir_swir"

OPATH         = "/workspaces/flowerstrips/ml/BB"
LABELS_FOLDER = os.path.join(OPATH, "Labels_filtered")
MODELFOLDER   = os.path.join(OPATH, "model")
ALL_RUNS_LOG  = os.path.join(MODELFOLDER, "all_runs.csv")
TEST_CHIPS_CSV = os.path.join(OPATH, "test_chips_list.csv")

CHIP_SIZE = 592
EPOCHS    = 200
PATIENCE  = 10            # Early stopping
POS_WEIGHT = 15.0
LR = 1e-3
VAL_FRACTION = 0.2
NUM_WORKERS = 0           # bewusst 0 für Reproduzierbarkeit
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
    raise ValueError(f"Unbekannte BAND_CONFIG: {BAND_CONFIG}")

NUM_CHANNELS = len(MONTHBANDS)

param_grid = {
    'weight_decay': [0, 1e-4, 1e-3],
    'batch_size':   [4, 8, 16],
}
PARAM_COMBINATIONS = list(ParameterGrid(param_grid))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Reproducibility — pro Run frisch aufrufen
# ============================================================
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Dataset  (1:1 aus Notebook Cell 15)
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

    def __len__(self): # Gibt Länge des Datensatzes zurück
        return len(self.image_paths)

    def __getitem__(self, idx): #Hier wird genau 1 Chip geliefert
        img = tifffile.imread(self.image_paths[idx])   # (H, W, C)
        lbl = tifffile.imread(self.label_paths[idx])   # (H, W)

        img = img[..., self.monthbands]
        h, w, c = img.shape
        if h < self.chip_size or w < self.chip_size:
            pad_h = max(0, self.chip_size - h)
            pad_w = max(0, self.chip_size - w)
            img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant", constant_values=0)
        img = img[:self.chip_size, :self.chip_size, :]

        img = img.astype(np.float32)
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
# Stratifizierter Split  (aus Notebook Cell 11)
# Test-CSV wird NUR gelesen, nie neu erzeugt.
# Split ist seed-abhängig -> einmal global bauen, für alle Runs identisch.
# ============================================================
def has_positive(label_path):
    with rasterio.open(label_path) as src:
        arr = src.read(1)
    return arr.sum() > 0


def build_split():
    set_seed(SEED)

    images_paths = sorted(glob(os.path.join(IMAGES_FOLDER, "*.tif")))
    labels_paths = sorted(glob(os.path.join(LABELS_FOLDER, "*.tif")))

    test_chips = pd.read_csv(TEST_CHIPS_CSV)
    test_chips_ls = test_chips["0"].to_list()
    test_set = set(test_chips_ls)

    train_pool = [(img, lbl) for img, lbl in zip(images_paths, labels_paths)
                  if os.path.basename(lbl) not in test_set]

    pos_pairs, neg_pairs = [], []
    for img, lbl in train_pool:
        (pos_pairs if has_positive(lbl) else neg_pairs).append((img, lbl))

    def split_train_val(pairs, val_frac):
        pairs = pairs.copy()
        random.shuffle(pairs)
        n_val = int(len(pairs) * val_frac)
        return pairs[n_val:], pairs[:n_val]

    pos_train, pos_val = split_train_val(pos_pairs, VAL_FRACTION)
    neg_train, neg_val = split_train_val(neg_pairs, VAL_FRACTION)

    train_pairs = pos_train + neg_train
    val_pairs   = pos_val + neg_val
    random.shuffle(train_pairs)
    random.shuffle(val_pairs)

    train_images, train_labels = map(list, zip(*train_pairs))
    val_images,   val_labels   = map(list, zip(*val_pairs))

    print(f"Train: {len(train_images)} | Val: {len(val_images)} | "
          f"Test: {len(test_set)} | Train pos: {len(pos_train)} | Val pos: {len(pos_val)}",
          flush=True)
    return train_images, train_labels, val_images, val_labels


# ============================================================
# IoU-Metrik (aus Notebook Cell 22)
# ============================================================
def iou_pytorch(outputs, labels, threshold=0.3):
    outputs = torch.sigmoid(outputs)
    outputs = (outputs > threshold).float()
    labels = (labels > 0.5).float()
    intersection = (outputs * labels).sum(dim=(1, 2, 3))
    union = ((outputs + labels) > 0).float().sum(dim=(1, 2, 3))
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()


# ============================================================
# Ein Trainingslauf
# ============================================================
def run_experiment(params, split):
    weight_decay = params["weight_decay"]
    batch_size   = params["batch_size"]

    model_name = (f"FS_unet_resnet34_pw{POS_WEIGHT:.0f}_bs{batch_size}_mergedALL"
                  f"_wd{weight_decay:.0e}_{BAND_CONFIG}")
    model_path = os.path.join(MODELFOLDER, f"{model_name}.pt")
    log_path   = os.path.join(MODELFOLDER, f"{model_name}.log")
    done_path  = os.path.join(MODELFOLDER, f"{model_name}.DONE")

    if os.path.exists(done_path):
        print(f"[skip] {model_name} (DONE existiert)", flush=True)
        return

    print(f"\n{'='*60}\n[start] {model_name}\n{'='*60}", flush=True)

    # --- Reproduzierbarkeit pro Run frisch ---
    set_seed(SEED)

    train_images, train_labels, val_images, val_labels = split
    train_ds = FlowerstripsDataset(train_images, train_labels, MONTHBANDS, NORM_COEF, CHIP_SIZE, augment=True)
    val_ds   = FlowerstripsDataset(val_images,   val_labels,   MONTHBANDS, NORM_COEF, CHIP_SIZE, augment=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True) # drop_last=False ist aktiv → alle Chips werden garantiert jede Epoche gesehen, unabhängig davon ob N durch 4 teilbar ist.
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # --- Modell / Optimizer / Scaler PRO RUN neu ---
    model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet",
                     in_channels=NUM_CHANNELS, classes=1).to(device)

    pos_weight = torch.tensor([POS_WEIGHT], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    scaler = GradScaler()

    # --- Log-Datei für diesen Run ---
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
                loss = criterion(outputs, labels)   # 1:1 wie Notebook (Loss im autocast)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * images.size(0)
            train_iou += iou_pytorch(outputs.detach().cpu(), labels.cpu()) * images.size(0)

        train_loss /= len(train_loader.dataset)
        train_iou  /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        tp_total = fp_total = fn_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)   # 1:1 wie Notebook
                val_loss += loss.item() * images.size(0)
                val_iou += iou_pytorch(outputs.cpu(), labels.cpu()) * images.size(0)

                pred_binary = (torch.sigmoid(outputs) > 0.3).float().cpu()
                labels_cpu = labels.cpu()
                tp_total += ((pred_binary == 1) & (labels_cpu == 1)).sum().item()
                fp_total += ((pred_binary == 1) & (labels_cpu == 0)).sum().item()
                fn_total += ((pred_binary == 0) & (labels_cpu == 1)).sum().item()

        val_loss /= len(val_loader.dataset)
        val_iou  /= len(val_loader.dataset)
        val_precision = tp_total / (tp_total + fp_total + 1e-6)
        val_recall    = tp_total / (tp_total + fn_total + 1e-6)

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

    # Sammeldatei (eine Zeile pro Run)
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
          f"{len(PARAM_COMBINATIONS)} Kombinationen", flush=True)

    if not os.path.exists(ALL_RUNS_LOG):
        with open(ALL_RUNS_LOG, "w", newline="") as f:
            csv.writer(f).writerow(["model_name", "date", "epoch", "train_loss",
                                    "val_loss", "train_iou", "val_iou",
                                    "val_precision", "val_recall"])

    split = build_split()

    for i, params in enumerate(PARAM_COMBINATIONS):
        print(f"\n### Grid {i+1}/{len(PARAM_COMBINATIONS)}: {params}", flush=True)
        run_experiment(params, split)

    print("\nAlle Runs dieser Band-Config abgeschlossen.", flush=True)


if __name__ == "__main__":
    main()
