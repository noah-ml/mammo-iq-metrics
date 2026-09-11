#!/usr/bin/env python3
"""Regenerate the headline results of the thesis from the released predictions.

No GPU, no model weights, no retraining. Everything here is computed from the
per-image prediction tables in the data record, each of which holds one row per
image x degradation x severity with the model logit and the image quality
metrics for that same variant.

Usage
-----
    python reproduce/reproduce_from_predictions.py \\
        --results-csv    data/convnext_nojitter_results_MASKEDSSIM_REFTAU.csv \\
        --multiarch-dir  data/multiarch \\
        --val-predictions data/val_clean_predictions.csv \\
        --out-dir        results/reproduced

The released tables have the VinDr-derived columns removed, because those are
dataset annotations and cannot be redistributed. Run ``join_labels.py`` first to
reattach them from your own credentialed copy of VinDr-Mammo.

With ``--check`` (the default) each reproduced quantity is compared against the
value reported in the thesis and the script exits non-zero if any of them drift
beyond tolerance.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

# Values as reported in the thesis, with the tolerance each is quoted to.
EXPECTED = {
    "clean_auc":                (0.824,  0.001),
    "clean_ap":                 (0.436,  0.001),
    "jpeg2000_s6_auc":          (0.654,  0.001),
    "jpeg2000_s6_ap":           (0.123,  0.001),
    "jpeg2000_s6_delta_auc":    (-0.170, 0.001),
    "ece_raw":                  (0.367,  0.001),
    "ece_platt":                (0.009,  0.002),
    "tau_clean_mean":           (0.0036, 0.0002),
    "tau_noise_s6_mean":        (0.0281, 0.0002),
    "ssim_shift_jpeg2000_rho":          (-0.821, 0.002),
    "ssim_shift_jpeg2000_within_sev_rho": (-0.331, 0.002),
    "bridge_ssim_auc_condition_rho":     (0.78,   0.005),
    "arch_mean_auc_convnext_tiny":   (0.828, 0.001),
    "arch_mean_auc_resnet18":        (0.753, 0.001),
    "arch_mean_auc_efficientnet_b4": (0.782, 0.001),
}

CLEAN = "clean"  # the released tables label the undegraded variant "clean"


def ece(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error over ``n_bins`` equally spaced bins.

    The thesis reports ECE with M = 15 equally spaced bins throughout. The
    uncalibrated value is insensitive to the bin count, but the recalibrated
    one is not: after Platt scaling the probabilities concentrate near the
    base rate, so a coarser grid merges bins that carry the residual error.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(prob, edges[1:-1]), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        total += m.mean() * abs(y_true[m].mean() - prob[m].mean())
    return float(total)


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = {"image_id", "label_45", "degradation_type", "severity", "logit", "prob"} - set(df.columns)
    if missing:
        sys.exit(
            f"{path.name} is missing {sorted(missing)}.\n"
            f"If the label columns are absent, run join_labels.py first."
        )
    return df


def condition_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """AUC and AP for every degradation x severity cell, plus change vs clean."""
    clean = df[df.degradation_type.eq(CLEAN)]
    auc0 = roc_auc_score(clean.label_45, clean.prob)
    ap0 = average_precision_score(clean.label_45, clean.prob)

    rows = []
    for (dtype, sev), g in df[~df.degradation_type.eq(CLEAN)].groupby(
        ["degradation_type", "severity"], sort=True
    ):
        auc = roc_auc_score(g.label_45, g.prob)
        ap = average_precision_score(g.label_45, g.prob)
        rows.append({
            "degradation_type": dtype, "severity": int(sev),
            "n": len(g), "n_pos": int(g.label_45.sum()),
            "auc": auc, "average_precision": ap,
            "delta_auc": auc - auc0, "delta_ap": ap - ap0,
        })
    out = pd.DataFrame(rows)
    out.attrs["clean_auc"] = auc0
    out.attrs["clean_ap"] = ap0
    return out


def prediction_shift(df: pd.DataFrame) -> pd.DataFrame:
    """Attach |logit_degraded - logit_clean| to every degraded observation."""
    base = (df[df.degradation_type.eq(CLEAN)]
            .set_index("image_id")["logit"].rename("logit_clean"))
    deg = df[~df.degradation_type.eq(CLEAN)].join(base, on="image_id")
    deg = deg.dropna(subset=["logit_clean"])
    deg["abs_shift"] = (deg["logit"] - deg["logit_clean"]).abs()
    return deg


def main() -> int:
    ap_ = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument("--results-csv", required=True, type=Path,
                     help="per-image predictions for the primary model")
    ap_.add_argument("--multiarch-dir", type=Path,
                     help="directory of *_degradation_results.csv for the 3x3 grid")
    ap_.add_argument("--val-predictions", type=Path,
                     help="clean validation predictions, needed for Platt scaling")
    ap_.add_argument("--out-dir", type=Path, default=Path("results/reproduced"))
    ap_.add_argument("--no-check", action="store_true",
                     help="report values without comparing against the thesis")
    a = ap_.parse_args()

    a.out_dir.mkdir(parents=True, exist_ok=True)
    got: dict[str, float] = {}

    # ---- primary model -----------------------------------------------------
    df = load(a.results_csv)
    cond = condition_metrics(df)
    cond.to_csv(a.out_dir / "condition_metrics.csv", index=False)
    got["clean_auc"] = cond.attrs["clean_auc"]
    got["clean_ap"] = cond.attrs["clean_ap"]

    jp6 = cond[cond.degradation_type.eq("jpeg2000") & cond.severity.eq(6)].iloc[0]
    got["jpeg2000_s6_auc"] = jp6.auc
    got["jpeg2000_s6_ap"] = jp6.average_precision
    got["jpeg2000_s6_delta_auc"] = jp6.delta_auc

    # ---- calibration -------------------------------------------------------
    clean = df[df.degradation_type.eq(CLEAN)]
    got["ece_raw"] = ece(clean.label_45.to_numpy(), clean.prob.to_numpy())

    if a.val_predictions is not None:
        val = pd.read_csv(a.val_predictions)
        lr = LogisticRegression(C=1e10, solver="lbfgs")
        lr.fit(val[["logit"]].to_numpy(), val["label_45"].to_numpy())
        p = lr.predict_proba(clean[["logit"]].to_numpy())[:, 1]
        got["ece_platt"] = ece(clean.label_45.to_numpy(), p)
        print(f"Platt scaling fitted on {len(val)} validation images: "
              f"a={lr.coef_[0][0]:.3f}, b={lr.intercept_[0]:.3f}")

    # ---- noise metric tau --------------------------------------------------
    if "tau" in df.columns:
        got["tau_clean_mean"] = float(clean.tau.mean())
        n6 = df[df.degradation_type.eq("noise") & df.severity.eq(6)]
        got["tau_noise_s6_mean"] = float(n6.tau.mean())

    # ---- RQ2: image quality metric against prediction shift ----------------
    if "ssim" in df.columns:
        deg = prediction_shift(df)

        # Per-degradation correlation between the metric and the size of the
        # prediction shift. These are the heatmap cells; SSIM under JPEG 2000
        # is the strongest of them.
        cells = []
        for (dtype, metric), rho in (
            ((d, m), spearmanr(g[m], g.abs_shift).statistic)
            for d, g in deg.groupby("degradation_type")
            for m in ("ssim", "tenengrad", "noise_var", "tau")
            if m in deg.columns
        ):
            cells.append({"degradation_type": dtype, "iqm": metric, "spearman_rho": rho})
        pd.DataFrame(cells).to_csv(a.out_dir / "iqm_shift_heatmap.csv", index=False)
        jp = deg[deg.degradation_type.eq("jpeg2000")]
        got["ssim_shift_jpeg2000_rho"] = float(spearmanr(jp.ssim, jp.abs_shift).statistic)

        # Within a fixed severity the anatomical spread between images replaces
        # the imposed spread, and the association weakens.
        per_cell = [spearmanr(g.ssim, g.abs_shift).statistic
                    for _, g in jp.groupby("severity") if g.ssim.nunique() > 1]
        got["ssim_shift_jpeg2000_within_sev_rho"] = float(np.mean(per_cell))

        # The bridge: one point per condition, condition-mean metric against the
        # AUC actually achieved under that condition.
        cm = (deg.groupby(["degradation_type", "severity"])["ssim"].mean()
              .rename("mean_ssim").reset_index()
              .merge(cond, on=["degradation_type", "severity"]))
        got["bridge_ssim_auc_condition_rho"] = float(
            spearmanr(cm.mean_ssim, cm.auc).statistic)
        cm.to_csv(a.out_dir / "bridge_condition_level.csv", index=False)

    # ---- cross-architecture grid ------------------------------------------
    if a.multiarch_dir is not None:
        per_file = {}
        for f in sorted(a.multiarch_dir.glob("*_degradation_results.csv")):
            d = pd.read_csv(f, usecols=["label_45", "degradation_type", "prob"])
            c = d[d.degradation_type.eq(CLEAN)]
            per_file[f.name] = roc_auc_score(c.label_45, c.prob)
        # the primary model is ConvNeXt seed 42 and may not live in that folder
        primary_is_listed = any("convnext" in k and "s42" in k for k in per_file)
        for arch in ("convnext_tiny", "resnet18", "efficientnet_b4"):
            vals = [v for k, v in per_file.items() if k.startswith(arch)]
            if arch == "convnext_tiny" and not primary_is_listed:
                vals.append(got["clean_auc"])
            if vals:
                got[f"arch_mean_auc_{arch}"] = float(np.mean(vals))
        pd.Series(per_file).to_csv(a.out_dir / "clean_auc_per_model.csv",
                                   header=["clean_auc"])

    # ---- report ------------------------------------------------------------
    print(f"\n{'quantity':<32} {'reproduced':>12} {'thesis':>10} {'status':>8}")
    print("-" * 66)
    failures = 0
    for name, value in got.items():
        if name in EXPECTED and not a.no_check:
            want, tol = EXPECTED[name]
            ok = abs(value - want) <= tol
            failures += (not ok)
            print(f"{name:<32} {value:12.4f} {want:10.4f} {'ok' if ok else 'DRIFT':>8}")
        else:
            print(f"{name:<32} {value:12.4f} {'':>10} {'':>8}")
    pd.Series(got).to_csv(a.out_dir / "headline_values.csv", header=["value"])
    print(f"\nwrote {a.out_dir}")

    if failures and not a.no_check:
        print(f"\n{failures} quantity/quantities drifted from the reported values.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
