#!/usr/bin/env python3
"""
Computes global_IoU for the best runs — exactly, via segmentation_models_pytorch,
from the raw TP/FP/FN sums stored during re-logging.

Background: the per-epoch val_iou is chip-averaged (micro-imagewise) and inflated
by the many empty chips: a correctly empty chip (TP=FP=FN=0) scores IoU=1.0
(zero_division=1.0). It is therefore NOT suitable as a comparison metric between
the spectral combinations.

global_IoU instead pools all pixels (micro): empty, correctly predicted chips
contribute 0 to TP/FP/FN and drop out of the computation:
    global_IoU = TP / (TP + FP + FN)   ==   smp.metrics.iou_score(..., "micro")

Input: all_runs_smp.csv (relog_all_runs_smp.py), which now holds the raw sums
val_tp/val_fp/val_fn/val_tn per best run. global_IoU is computed directly from
them via smp — full precision, same implementation as compute_test_metrics.py
(global_iou, micro), no reconstruction from rounded P/R.

Fallback: if the sum columns are missing (older CSV), global_IoU is reconstructed
algebraically from P/R (1/(1/P + 1/R - 1)) — mathematically identical but only as
accurate as the rounded P/R values. A warning is printed.

Result is saved as all_runs_smp_with_iou.csv.
"""

import os
import pandas as pd
import torch
import segmentation_models_pytorch as smp

MODEL_DIR = "/workspaces/flowerstrips/ml/BB/model"
RUNS_CSV = os.path.join(MODEL_DIR, "all_runs_smp.csv")
OUT_CSV  = os.path.join(MODEL_DIR, "all_runs_smp_with_iou.csv")

RAW_COLS = ["val_tp", "val_fp", "val_fn", "val_tn"]


def global_iou_from_counts(tp, fp, fn, tn):
    """Exact, via smp.metrics.iou_score(reduction='micro') from raw counters.
    Identical to TP/(TP+FP+FN); same implementation as the test script."""
    t = torch.tensor([[int(tp)]]); f = torch.tensor([[int(fp)]])
    n = torch.tensor([[int(fn)]]); v = torch.tensor([[int(tn)]])
    return float(smp.metrics.iou_score(t, f, n, v, reduction="micro"))


def global_iou_from_pr(precision, recall):
    """Fallback: algebraic reconstruction 1/(1/P + 1/R - 1)."""
    if precision <= 0 or recall <= 0:
        return 0.0
    return 1.0 / (1.0 / precision + 1.0 / recall - 1.0)


df = pd.read_csv(RUNS_CSV)

have_counts = all(c in df.columns for c in RAW_COLS)
if have_counts:
    # compute exactly only for rows with complete (non-NaN) sums
    def _giou(r):
        if all(pd.notna(r[c]) for c in RAW_COLS):
            return global_iou_from_counts(r["val_tp"], r["val_fp"],
                                          r["val_fn"], r["val_tn"])
        # single row without sums -> fallback
        return global_iou_from_pr(r["val_precision"], r["val_recall"])
    df["global_iou"] = df.apply(_giou, axis=1)
    n_exact = int(df[RAW_COLS].notna().all(axis=1).sum())
    print(f"global_iou via smp from raw TP/FP/FN: {n_exact}/{len(df)} rows exact.")
    if n_exact < len(df):
        print(f"  {len(df) - n_exact} rows without sums -> algebraic fallback "
              f"(only one band config re-logged?).")
else:
    print("WARNING: sum columns (val_tp/fp/fn/tn) missing in the CSV -> "
          "global_iou reconstructed algebraically from rounded P/R. "
          "For full precision, re-run relog_all_runs_smp.py (new version).")
    df["global_iou"] = df.apply(
        lambda r: global_iou_from_pr(r["val_precision"], r["val_recall"]), axis=1)

# column order: global_iou directly after val_recall
cols = list(df.columns)
cols.insert(cols.index("val_recall") + 1, cols.pop(cols.index("global_iou")))
df = df[cols]

df.to_csv(OUT_CSV, index=False)
print(f"Saved: {OUT_CSV}\n")

# sorted by global_IoU -> winner per spectral combination is easy to read off
view = df[["model_name", "val_iou", "val_precision", "val_recall", "global_iou"]]
print(view.sort_values("global_iou", ascending=False).to_string(index=False))