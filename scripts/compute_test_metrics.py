#!/usr/bin/env python3
"""
compute_test_metrics.py
───────────────────────
Test metrics for the tuned transfer model (or any list of models)
on the evaluation chips.

IMPORTANT — Normalisation/loading is carried out exactly as in train.py:
  The same FlowerstripsDataset is imported. This ensures that
  channel slicing (MONTHBANDS), normalisation (NO clip), padding and
  transpose are guaranteed to be bit-identical to the training and threshold tuning.

  Aggregation levels (for the binary case, in smp micro==macro==weighted;
  the image level is controlled via the ‘imagewise’ suffix):
    - global_*      = reduction="micro"
                      -> tp/fp/fn/tn pooled across ALL pixels of ALL chips,
                         then ONE score. This is the primary metric (invariant
                         with respect to the ~90% empty chips). Corresponds to the old
                         “global_iou”.
    - mean_* (chip) = reduction="micro-imagewise", zero_division=1.0
                      -> Score PER chip, then averaged across chips. zero_division=1.0
                         assigns the value 1.0 to correctly empty chips (tp=fp=fn=0) —
                         the same convention as the old 1e-6 smoothing formula
                         and as the (converted to SMP) `val_iou` in `train.py`.
                         INFLATED by the ~90% empty chips, for diagnostic purposes only.

Distance metric:
  - HD95 (95th percentile of the symmetric surface Hausdorff distance) via
    medpy.metric.binary.hd95 — edge-based and outlier-robust (replaces the
    previous outlier-sensitive custom implementation of max-Hausdorff).
  - Defined only on chips with a non-empty prediction AND a non-empty label
    (otherwise medpy raises a RuntimeError) -> otherwise NaN -> conditioning on TP chips.
  - voxelspacing=PIXEL_SIZE_M -> HD95 directly in metres.

Run discipline:
  - During development: EVAL_CSV = val_chips_list.csv
  - Final run ONCE with test_chips_list.csv; do not make any further adjustments afterwards.

Threshold:
  - Transfer model (RGB+NIR bs16_wd1e-04):      THRESHOLD = 0.65 (tuned to Val)
  - Transfer model (RGB+NIR+SWIR bs8_wd0e+00): THRESHOLD = 0.70 (tuned to Val)

    nohup python -u compute_test_metrics.py > test_metrics.log 2>&1 &
"""

import os
import numpy as np
import pandas as pd
import torch
import segmentation_models_pytorch as smp
from pathlib import Path
from medpy.metric.binary import hd95 as medpy_hd95
from torch.utils.data import DataLoader
from tqdm import tqdm


from train import (
    FlowerstripsDataset,
    build_split,
    MONTHBANDS,
    NORM_COEF,
    CHIP_SIZE,
    NUM_CHANNELS,
    NUM_WORKERS,
    MODELFOLDER,
    LABELS_FOLDER,
    IMAGES_FOLDER,
    OPATH,
    BAND_CONFIG,
    TEST_CHIPS_CSV,
    VAL_CHIPS_CSV,
    device,
)

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION 
# ──────────────────────────────────────────────────────────────────────────────

# ── Eval-Set: during the VAL development phase, final TEST run ────────────────────
#EVAL_CSV = VAL_CHIPS_CSV          
EVAL_CSV = TEST_CHIPS_CSV          

THRESHOLD = 0.65

# ── Model
MODEL_PATHS = [
    str(Path(MODELFOLDER) / "FS_unet_resnet34_pw15_bs16_wd1e-04_rgb_nir.pt"), 
]

PER_CHIP_DIR = os.path.join(OPATH, "per_chip")
SUMMARY_CSV  = os.path.join(OPATH, "summary_per_model.csv")
ENCODER      = "resnet34"
NUM_CLASSES  = 1
PIXEL_SIZE_M = 2.5   # for HD95 conversion: px to m (voxel spacing)

os.makedirs(PER_CHIP_DIR, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# EVAL-SPLIT als (image_paths, label_paths)
# ──────────────────────────────────────────────────────────────────────────────
def build_eval_split(eval_csv):
    names = pd.read_csv(eval_csv).iloc[:, 0].dropna().tolist()
    img_paths = [os.path.join(IMAGES_FOLDER, n) for n in names]
    lbl_paths = [os.path.join(LABELS_FOLDER, n) for n in names]
    # nur existierende Paare behalten, Reihenfolge stabil
    keep = [(i, l, n) for i, l, n in zip(img_paths, lbl_paths, names)
            if os.path.exists(i) and os.path.exists(l)]
    img_paths = [k[0] for k in keep]
    lbl_paths = [k[1] for k in keep]
    names     = [k[2] for k in keep]
    return img_paths, lbl_paths, names


def load_model(model_path):
    model = smp.Unet(encoder_name=ENCODER, encoder_weights=None,
                     in_channels=NUM_CHANNELS, classes=NUM_CLASSES,
                     activation=None)
    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


# ──────────────────────────────────────────────────────────────────────────────
# HD95 (randbasiert, outlier-robust) via medpy.
# medpy expects result=Prediction, reference=Label; raises a RuntimeError if
# either of the two masks is empty -> catch -> NaN (TP chip conditioning).
# voxelspacing=PIXEL_SIZE_M -> result in metres.
# ──────────────────────────────────────────────────────────────────────────────
def hd95_single(pred_bin, label_bin, spacing=PIXEL_SIZE_M):
    pred  = pred_bin.astype(bool)
    label = label_bin.astype(bool)
    if not pred.any() or not label.any():
        return np.nan                   # mind. eine Maske leer -> undefiniert
    try:
        return float(medpy_hd95(pred, label, voxelspacing=spacing))
    except RuntimeError:
        return np.nan                   # Sicherheitsnetz


# ──────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ──────────────────────────────────────────────────────────────────────────────
def evaluate_model(model_path, img_paths, lbl_paths, chip_names):
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    print(f"\n{'='*70}\nModell: {model_name}  (THRESHOLD={THRESHOLD})\n{'='*70}",
          flush=True)

    if not os.path.exists(model_path):
        print(f"  Modell fehlt: {model_path}, ueberspringe.", flush=True)
        return None

    model = load_model(model_path)

    # Dataset EXACTLY as in training (augment=False), deterministic order
    ds = FlowerstripsDataset(img_paths, lbl_paths, MONTHBANDS, NORM_COEF,
                             CHIP_SIZE, augment=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)

    # Collect pro-chip tp/fp/fn/tn as tensors (for SMP reductions).
    # Shape per chip: (1, 1) → stacked (N, 1) as expected by SMP for binary.
    tp_list, fp_list, fn_list, tn_list = [], [], [], []
    rows = []

    with torch.no_grad():
        for idx, (img, lbl) in enumerate(tqdm(loader, desc="Inference")):
            img = img.to(device, non_blocking=True)
            logits = model(img)
            prob = torch.sigmoid(logits)

            # smp.get_stats: output float (N,1,H,W) + threshold, target long (N,1,H,W)
            target = (lbl > 0.5).long().to(device)
            if target.dim() == 3:            # (N,H,W) -> (N,1,H,W)
                target = target.unsqueeze(1)
            if prob.dim() == 3:
                prob = prob.unsqueeze(1)

            tp, fp, fn, tn = smp.metrics.get_stats(
                prob, target, mode="binary", threshold=THRESHOLD)
            # tp.. shape (N=1, C=1); collect on CPU
            tp_list.append(tp.cpu()); fp_list.append(fp.cpu())
            fn_list.append(fn.cpu()); tn_list.append(tn.cpu())

            tp_i = int(tp.sum().item()); fp_i = int(fp.sum().item())
            fn_i = int(fn.sum().item()); tn_i = int(tn.sum().item())

            # Per-chip scores via SMP (imagewise, zero_division=1.0), so that
            # the per-chip columns follow exactly the same convention as the
            # mean_* (micro-imagewise) that will be aggregated later. Single-chip reduction:
            t1 = tp.cpu(); f1_ = fp.cpu(); n1 = fn.cpu(); z1 = tn.cpu()
            iou_i  = float(smp.metrics.iou_score(t1, f1_, n1, z1,
                        reduction="micro-imagewise", zero_division=1.0))
            prec_i = float(smp.metrics.precision(t1, f1_, n1, z1,
                        reduction="micro-imagewise", zero_division=1.0))
            rec_i  = float(smp.metrics.recall(t1, f1_, n1, z1,
                        reduction="micro-imagewise", zero_division=1.0))
            f1_i   = float(smp.metrics.f1_score(t1, f1_, n1, z1,
                        reduction="micro-imagewise", zero_division=1.0))

            pred_np  = (prob.squeeze() > THRESHOLD).cpu().numpy().astype(bool)
            label_np = target.squeeze().cpu().numpy().astype(bool)
            hd = hd95_single(pred_np, label_np)

            rows.append({
                "chip":           chip_names[idx],
                # RAW counters — required for paired Bootstrap (not rounded!)
                "tp": tp_i, "fp": fp_i, "fn": fn_i, "tn": tn_i,
                "iou":            round(iou_i,  6),
                "precision":      round(prec_i, 6),
                "recall":         round(rec_i,  6),
                "f1":             round(f1_i,   6),
                "hd95_m":         round(hd, 4) if not np.isnan(hd) else np.nan,
                "label_pos_frac": round(float(label_np.sum()) / label_np.size, 6),
                "pred_pos_frac":  round(float(pred_np.sum())  / pred_np.size,  6),
            })

    if not rows:
        print("  No valid tokens.", flush=True)
        return None

    per_chip_df = pd.DataFrame(rows)
    per_chip_path = os.path.join(PER_CHIP_DIR, f"{model_name}_thr{THRESHOLD}.csv")
    per_chip_df.to_csv(per_chip_path, index=False)
    print(f"  Pro-Chip CSV: {per_chip_path}", flush=True)

    # ── Aggregation via SMP from the stacked per-chip counters ───────────
    TP = torch.cat(tp_list, dim=0)   # (N, 1)
    FP = torch.cat(fp_list, dim=0)
    FN = torch.cat(fn_list, dim=0)
    TN = torch.cat(tn_list, dim=0)

    # PRIMAERMASS: micro = pooled across all pixels on all chips.
    global_iou  = float(smp.metrics.iou_score(TP, FP, FN, TN, reduction="micro"))
    global_prec = float(smp.metrics.precision(TP, FP, FN, TN, reduction="micro"))
    global_rec  = float(smp.metrics.recall(TP, FP, FN, TN, reduction="micro"))
    global_f1   = float(smp.metrics.f1_score(TP, FP, FN, TN, reduction="micro"))

    # DIAGNOSTISCH: micro-imagewise = score per chip, then average; empty chips = 1.0.
    mean_iou  = float(smp.metrics.iou_score(TP, FP, FN, TN,
                    reduction="micro-imagewise", zero_division=1.0))
    mean_prec = float(smp.metrics.precision(TP, FP, FN, TN,
                    reduction="micro-imagewise", zero_division=1.0))
    mean_rec  = float(smp.metrics.recall(TP, FP, FN, TN,
                    reduction="micro-imagewise", zero_division=1.0))
    mean_f1   = float(smp.metrics.f1_score(TP, FP, FN, TN,
                    reduction="micro-imagewise", zero_division=1.0))

    # HD95 only for positive chips (label_pos_frac > 0) AND where defined
    # (non-empty prediction → hd95_m not NaN). This excludes the ~90% of empty
    # chips that would otherwise push the median down to 0.
    pos_mask = per_chip_df["label_pos_frac"] > 0
    n_pos_chips = int(pos_mask.sum())
    hd95_valid = per_chip_df.loc[pos_mask, "hd95_m"].dropna()
    median_hd95 = float(hd95_valid.median()) if len(hd95_valid) else np.nan
    mean_hd95   = float(hd95_valid.mean())   if len(hd95_valid) else np.nan

    summary = {
        "model":            model_name,
        "band_config":      BAND_CONFIG,
        "threshold":        THRESHOLD,
        "eval_set":         os.path.basename(EVAL_CSV),
        "n_chips":          len(per_chip_df),
        # PRIMAER (micro / gepoolt)
        "global_iou":       round(global_iou,  6),
        "global_precision": round(global_prec, 6),
        "global_recall":    round(global_rec,  6),
        "global_f1":        round(global_f1,   6),
        # DIAGNOSTISCH (micro-imagewise / per-Chip-Mittel, leere Chips=1.0)
        "mean_iou":         round(mean_iou,  6),
        "mean_precision":   round(mean_prec, 6),
        "mean_recall":      round(mean_rec,  6),
        "mean_f1":          round(mean_f1,   6),
        # HD95 (nur Positiv-Chips mit definierter Distanz, in Metern)
        "n_pos_chips":      n_pos_chips,
        "n_hd95_valid":     int(len(hd95_valid)),
        "median_hd95_m":    round(median_hd95, 4) if not np.isnan(median_hd95) else np.nan,
        "mean_hd95_m":      round(mean_hd95,   4) if not np.isnan(mean_hd95)   else np.nan,
    }
    print(f"  [micro]  global_iou={global_iou:.4f}  P={global_prec:.4f}  "
          f"R={global_rec:.4f}  F1={global_f1:.4f}", flush=True)
    print(f"  [imgwise] mean_iou={mean_iou:.4f}  P={mean_prec:.4f}  "
          f"R={mean_rec:.4f}  F1={mean_f1:.4f}", flush=True)
    print(f"  [HD95]   median={median_hd95:.3f} m  mean={mean_hd95:.3f} m  "
          f"(n_valid={len(hd95_valid)}/{n_pos_chips} pos-chips)", flush=True)

    # ── SANITY CHECK (only relevant when EVAL_CSV=VAL & THRESHOLD=0.3) ───────────
    # The per-chip mean_iou (micro-imagewise, zero_division=1.0) MUST now correspond to the
    # val_iou logged in train.py (also SMP, same reduction,
    # threshold=0.3) for the same checkpoint — apart from fp16/fp32-
    # rounding. If it deviates significantly, the threshold, checkpoint or
    # chip list is incorrect.

    if os.path.basename(EVAL_CSV) == os.path.basename(VAL_CHIPS_CSV):
        print(f"  [SANITY] mean_iou={mean_iou:.5f} should match train.py val_iou (t=0.3, "
              f"smp micro-imagewise) for this checkpoint. Current "
              f"THRESHOLD={THRESHOLD} "
              f"{'(OK for comparison)' if THRESHOLD == 0.3 else '(!! set comparison to 0.3)'}",
              flush=True)
    return summary


def main():
    img_paths, lbl_paths, chip_names = build_eval_split(EVAL_CSV)
    print(f"Band-Config: {BAND_CONFIG} | IN_CHANNELS={NUM_CHANNELS} | "
          f"Eval: {os.path.basename(EVAL_CSV)} | Chips: {len(chip_names)} | "
          f"THRESHOLD={THRESHOLD} | Device: {device}", flush=True)

    summaries = []
    for mp in MODEL_PATHS:
        s = evaluate_model(mp, img_paths, lbl_paths, chip_names)
        if s is not None:
            summaries.append(s)

    if not summaries:
        print("\nNo models were analysed.", flush=True)
        return

    summary_df = pd.DataFrame(summaries)
    if os.path.exists(SUMMARY_CSV):
        old = pd.read_csv(SUMMARY_CSV)
        # Overwrite the same (model, threshold, eval_set) combination rather than duplicating it
        if {"model", "threshold", "eval_set"}.issubset(old.columns):
            mask = ~old.set_index(["model", "threshold", "eval_set"]).index.isin(
                summary_df.set_index(["model", "threshold", "eval_set"]).index)
            old = old[mask.values]
        summary_df = pd.concat([old, summary_df], ignore_index=True)

    summary_df = summary_df.sort_values("global_iou", ascending=False)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    print(f"\nSummary saved: {SUMMARY_CSV}", flush=True)
    print(summary_df[["model", "band_config", "threshold", "eval_set",
                      "global_iou", "global_f1", "global_precision",
                      "global_recall"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()