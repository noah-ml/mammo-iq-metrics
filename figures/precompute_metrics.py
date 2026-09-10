#!/usr/bin/env python3
"""Precompute the per-condition metric table (no inference; uses 957382 CSV).
Outputs tables/condition_metrics_full.csv and a markdown version."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

import pubcommon as pc


def main():
    R = pc.load_results()
    thr = pc.clean_youden_threshold(R)
    print(f"clean Youden threshold = {thr:.4f}")

    rows = []
    for dtype, sev in pc.conditions():
        sub = R[(R.degradation_type == dtype) & (R.severity == sev)]
        y = sub.label_45.to_numpy(); p = sub.prob.to_numpy()
        auc = roc_auc_score(y, p); ap = average_precision_score(y, p)
        auc_lo, auc_hi = pc.boot_ci_metric(y, p, roc_auc_score)
        ap_lo, ap_hi = pc.boot_ci_metric(y, p, average_precision_score)
        tm = pc.threshold_metrics(y, p, thr)
        rows.append(dict(
            degradation_type=dtype, severity=sev, n=len(sub), n_pos=int(y.sum()),
            auc=auc, auc_lo=auc_lo, auc_hi=auc_hi,
            ap=ap, ap_lo=ap_lo, ap_hi=ap_hi,
            ece=pc.expected_calibration_error(y, p), brier=pc.brier(y, p),
            **tm,
            mean_ssim=sub.ssim.mean(), mean_tenengrad=sub.tenengrad.mean(),
            mean_noise_var=sub.noise_var.mean(), mean_tau=sub.tau.mean(),
            median_ssim=sub.ssim.median(), median_tenengrad=sub.tenengrad.median(),
            median_noise_var=sub.noise_var.median(), median_tau=sub.tau.median(),
        ))
    df = pd.DataFrame(rows)
    # deltas vs clean
    clean = df[df.degradation_type == "clean"].iloc[0]
    df["delta_auc"] = df.auc - clean.auc
    df["delta_ap"] = df.ap - clean.ap
    df.to_csv(pc.TABLES / "condition_metrics_full.csv", index=False)

    # markdown
    cols = ["degradation_type", "severity", "auc", "ap", "sensitivity", "specificity",
            "f1", "balanced_accuracy", "ece", "brier"]
    md = df[cols].copy()
    for c in ["auc", "ap", "sensitivity", "specificity", "f1", "balanced_accuracy", "ece", "brier"]:
        md[c] = md[c].map(lambda v: f"{v:.4f}")
    (pc.TABLES / "condition_metrics_full.md").write_text(
        "# Per-condition metrics (ConvNeXt-Tiny no-ColorJitter, run 957382)\n\n"
        f"Fixed threshold = clean Youden = {thr:.4f}. n={pc.N_TEST}, positives={pc.N_POS}.\n\n"
        + md.to_markdown(index=False), encoding="utf-8")
    print("wrote tables/condition_metrics_full.csv + .md  (clean AUC=%.4f AP=%.4f)" %
          (clean.auc, clean.ap))


if __name__ == "__main__":
    main()
