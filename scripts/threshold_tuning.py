#!/usr/bin/env python3
# ============================================================
# threshold_tuning.py — Threshold sweep on Val (RGB+NIR model)
#
# Runs ONLY on Val. Model weights remain unchanged.
# Does not collect prob vectors -> RAM-safe (incremental TP/FP/FN).
# Best threshold based on global_IoU -> goes directly into test inference + NRW transfer.
#
# IMPORTANT: BAND_CONFIG in train.py must be set to "rgb_nir", as this
# script imports MONTHBANDS / NORM_COEF / IMAGES_FOLDER from there.
#
# Start:
#   nohup python -u threshold_tuning.py > threshold_tuning.log 2>&1 &
#   tail -f threshold_tuning.log
# ============================================================

import os
import torch
import numpy as np
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader

from train import (
    FlowerstripsDataset,
    build_split,
    MONTHBANDS,
    NORM_COEF,
    CHIP_SIZE,
    NUM_CHANNELS,
    NUM_WORKERS,
    MODELFOLDER,
    BAND_CONFIG,
    device,
)

MODEL_PATH = os.path.join(
    MODELFOLDER,
    "FS_unet_resnet34_pw15_bs8_wd0e+00_rgb_nir_swir.pt"
)

BATCH_SIZE = 8   # merely an inference; the value is irrelevant (does not affect the result)

# Note: this script applies to the rgb_nir transfer model
#assert BAND_CONFIG == "rgb_nir", (
   #f"BAND_CONFIG in train.py is '{BAND_CONFIG}', expected 'rgb_nir'. "
    #f"Threshold tuning only runs on the RGB+NIR transfer model."
#)


def main():
    # 1) load Val-Split
    train_images, train_labels, val_images, val_labels = build_split()

    val_ds = FlowerstripsDataset(val_images, val_labels, MONTHBANDS,
                                 NORM_COEF, CHIP_SIZE, augment=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    # 2) Build model + load custom weights (encoder_weights=None:
    #    no ImageNet required; we’ll overwrite it with the checkpoint anyway)
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                     in_channels=NUM_CHANNELS, classes=1).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"Modell geladen: {MODEL_PATH}", flush=True)

    # 3) Incremental threshold sweep (RAM-safe)
    #    No accumulation of prob vectors -> only 3*len(thresholds) int64 counters.
    #    fp32 (no autocast) for a clean decision boundary.
    thresholds = np.arange(0.05, 1.0, 0.05)
    tp = np.zeros(len(thresholds), dtype=np.int64)
    fp = np.zeros(len(thresholds), dtype=np.int64)
    fn = np.zeros(len(thresholds), dtype=np.int64)

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            probs = torch.sigmoid(model(images)).cpu().numpy().ravel()
            lab = labels.numpy().ravel().astype(bool)
            n_pos = lab.sum()
            for i, t in enumerate(thresholds):
                pred = probs >= t
                tp_i = np.count_nonzero(pred & lab)
                pred_pos = np.count_nonzero(pred)
                tp[i] += tp_i
                fp[i] += pred_pos - tp_i    # Predicted positives minus true positives
                fn[i] += n_pos - tp_i       # actual positives minus hits

    # 4) global_IoU pro Threshold (gepoolt: TP/(TP+FP+FN))
    print("\nthr    global_IoU        TP           FP           FN", flush=True)
    results = []
    for i, t in enumerate(thresholds):
        denom = tp[i] + fp[i] + fn[i]
        giou = tp[i] / denom if denom > 0 else 0.0
        prec = tp[i] / (tp[i] + fp[i]) if (tp[i] + fp[i]) > 0 else 0.0
        rec  = tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) > 0 else 0.0
        results.append((t, giou, prec, rec))
        print(f"{t:.2f}   {giou:.4f}   {tp[i]:>11} {fp[i]:>11} {fn[i]:>11}"
              f"   P={prec:.3f} R={rec:.3f}", flush=True)

    best_t, best_giou, _, _ = max(results, key=lambda r: r[1])
    print(f"\nBest threshold by global_IoU (Val): {best_t:.2f} "
          f"(global_IoU={best_giou:.4f})", flush=True)


if __name__ == "__main__":
    main()