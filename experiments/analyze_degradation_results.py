#!/usr/bin/env python3
"""
analyze_degradation_results.py
==============================
Post-processing for evaluate_degradations.py output. Computes the RQ1
(degradation robustness) and RQ2 (IQM <-> performance correlation) analyses
sketched in eval_pipeline_notes.md ("After cluster run — analysis scripts
needed"), starting from a single <model>_degradation_results.csv.

RQ1 — degradation robustness
    For each (degradation_type, severity): AUC on (prob, label_45), plus
    sensitivity at a FIXED operating threshold derived once from the clean
    condition's ROC curve via Youden's J statistic (mirrors the "fixed ROI"
    principle — the threshold is determined on the undegraded baseline and
    reused everywhere for a fair comparison). Reports, per degradation type,
    the first severity at which AUC drops more than 5% relative to the clean
    baseline ("threshold severity").

RQ2 — IQM <-> performance correlation
    Per (image, condition): delta_logit = logit(degraded) - logit(clean).
    Spearman rho between each IQM (noise_var, tenengrad, ssim, cnr, delta_mu, tau)
    and |delta_logit|, pooled over severities 1-6, separately per degradation
    type (the clean condition is excluded — delta_logit is identically 0
    there by construction).

Outputs (written under --output-dir, prefixed with --model-name):
    <model>_severity_metrics.csv     — AUC / sensitivity / rel. AUC drop per (type, severity)
    <model>_threshold_summary.csv    — first severity with >5% AUC drop, per degradation type
    <model>_iqm_correlations.csv     — Spearman rho / p-value per (type, IQM)
    <model>_auc_vs_severity.png
    <model>_sensitivity_vs_severity.png
    <model>_iqm_correlation.png

Usage
-----
    python analyze_degradation_results.py \
        --results-csv data/convnext_nojitter_results_MASKEDSSIM_REFTAU.csv \
        --output-dir  results/analysis_convnext \
        --model-name  convnext_tiny
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, roc_curve

DEGRADATION_TYPES = ["noise", "blur", "contrast", "jpeg2000", "resolution"]
IQM_COLS = ["noise_var", "tenengrad", "ssim", "cnr", "delta_mu", "tau"]
AUC_DROP_THRESHOLD = 0.05  # relative AUC drop that defines the "threshold severity"

log = logging.getLogger("analyze_degradation_results")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_results(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    before = len(df)
    df = df.drop_duplicates(subset=["image_id", "degradation_type", "severity"])
    if len(df) < before:
        log.info(f"  Dropped {before - len(df)} duplicate (image_id, degradation_type, severity) rows")

    df["severity"] = pd.to_numeric(df["severity"], errors="coerce").astype(int)
    for c in ["label_45", "label_5", "logit", "prob"] + IQM_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# RQ1 — AUC & sensitivity vs. severity
# ---------------------------------------------------------------------------

def youden_threshold(y_true: pd.Series, y_prob: pd.Series) -> float:
    """Operating threshold maximizing TPR - FPR on the clean condition."""
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    return float(thr[int(np.argmax(tpr - fpr))])


def compute_severity_metrics(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    clean = df[df["degradation_type"] == "clean"]
    baseline_auc = float(roc_auc_score(clean["label_45"], clean["prob"]))
    thresh = youden_threshold(clean["label_45"], clean["prob"])
    log.info(f"Baseline (clean) AUC = {baseline_auc:.4f}")
    log.info(f"Fixed operating threshold (Youden's J on clean ROC) = {thresh:.4f}")

    conditions = [("clean", 0)] + [(d, s) for d in DEGRADATION_TYPES for s in range(1, 7)]
    rows = []
    for deg_type, severity in conditions:
        sub = df[(df["degradation_type"] == deg_type) & (df["severity"] == severity)]
        auc = float(roc_auc_score(sub["label_45"], sub["prob"])) if sub["label_45"].nunique() > 1 else float("nan")
        pos = sub["label_45"] == 1
        pred_pos = sub["prob"] >= thresh
        sensitivity = float((pred_pos & pos).sum() / pos.sum()) if pos.sum() > 0 else float("nan")
        rel_drop = (baseline_auc - auc) / baseline_auc if baseline_auc else float("nan")
        rows.append({
            "degradation_type": deg_type,
            "severity": severity,
            "n_images": int(len(sub)),
            "auc": auc,
            "sensitivity": sensitivity,
            "auc_relative_drop_vs_clean": rel_drop,
        })
    return pd.DataFrame(rows), baseline_auc


def compute_threshold_severities(metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for deg_type in DEGRADATION_TYPES:
        sub = metrics_df[(metrics_df["degradation_type"] == deg_type) & (metrics_df["severity"] > 0)]
        sub = sub.sort_values("severity")
        hit = sub[sub["auc_relative_drop_vs_clean"] > AUC_DROP_THRESHOLD]
        first = int(hit.iloc[0]["severity"]) if len(hit) else None
        rows.append({"degradation_type": deg_type, "first_severity_auc_drop_gt_5pct": first})
        if first is not None:
            log.info(f"  {deg_type:<12} AUC first drops > {AUC_DROP_THRESHOLD:.0%} at severity {first}")
        else:
            log.info(f"  {deg_type:<12} AUC never drops > {AUC_DROP_THRESHOLD:.0%} across severities 1-6")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# RQ2 — IQM <-> |delta_logit| correlation
# ---------------------------------------------------------------------------

def compute_iqm_correlations(df: pd.DataFrame) -> pd.DataFrame:
    clean_logit = (
        df[df["degradation_type"] == "clean"][["image_id", "logit"]]
        .drop_duplicates(subset=["image_id"])
        .rename(columns={"logit": "logit_clean"})
    )
    merged = df[df["degradation_type"] != "clean"].merge(clean_logit, on="image_id", how="left")
    merged["abs_delta_logit"] = (merged["logit"] - merged["logit_clean"]).abs()

    rows = []
    for deg_type in DEGRADATION_TYPES:
        sub = merged[merged["degradation_type"] == deg_type]
        for metric in IQM_COLS:
            if metric not in sub.columns:
                continue
            valid = sub[[metric, "abs_delta_logit"]].dropna()
            if len(valid) < 10:
                rho, p = float("nan"), float("nan")
            else:
                rho, p = (float(v) for v in spearmanr(valid[metric], valid["abs_delta_logit"]))
            rows.append({
                "degradation_type": deg_type,
                "iqm": metric,
                "n_pairs": int(len(valid)),
                "spearman_rho": rho,
                "p_value": p,
            })

    out = pd.DataFrame(rows)
    log.info("Spearman rho(IQM, |delta_logit|) pooled over severities 1-6, per degradation type:")
    for _, r in out.iterrows():
        if np.isnan(r["spearman_rho"]):
            log.info(f"    {r['degradation_type']:<12} {r['iqm']:<12} n={r['n_pairs']:<6} rho=   n/a (insufficient non-null pairs)")
        else:
            log.info(f"    {r['degradation_type']:<12} {r['iqm']:<12} n={r['n_pairs']:<6} rho={r['spearman_rho']:+.3f}  p={r['p_value']:.2e}")
    return out


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_vs_severity(metrics_df: pd.DataFrame, baseline_auc: float, value_col: str,
                     ylabel: str, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for deg_type in DEGRADATION_TYPES:
        sub = metrics_df[(metrics_df["degradation_type"] == deg_type) & (metrics_df["severity"] > 0)]
        sub = sub.sort_values("severity")
        ax.plot(sub["severity"], sub[value_col], marker="o", label=deg_type)
    if value_col == "auc":
        ax.axhline(baseline_auc, color="black", linestyle="--", linewidth=1, label="clean baseline")
    ax.set_xlabel("Severity")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(range(1, 7))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_iqm_correlation(corr_df: pd.DataFrame, out_path: Path) -> None:
    pivot = corr_df.pivot(index="iqm", columns="degradation_type", values="spearman_rho")
    pivot = pivot.reindex(index=IQM_COLS, columns=DEGRADATION_TYPES)
    fig, ax = plt.subplots(figsize=(9, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Spearman rho  (IQM vs. |delta_logit|)")
    ax.set_title("IQM <-> model-sensitivity correlation per degradation type")
    ax.legend(fontsize=8, title="degradation")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-csv", type=str, required=True, help="<model>_degradation_results.csv from evaluate_degradations.py")
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--model-name", type=str, default=None, help="Output-file prefix (default: derived from --results-csv filename)")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    results_csv = Path(args.results_csv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.model_name or results_csv.stem.replace("_degradation_results", "")

    log.info(f"Loading {results_csv} ...")
    df = load_results(results_csv)
    log.info(f"  {len(df):,} rows | {df['image_id'].nunique():,} images | "
             f"degradation types present: {sorted(df['degradation_type'].unique())}")

    log.info("\n--- RQ1: AUC & sensitivity vs. severity ---")
    metrics_df, baseline_auc = compute_severity_metrics(df)
    metrics_path = out_dir / f"{name}_severity_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    log.info(f"Wrote {metrics_path}")

    threshold_df = compute_threshold_severities(metrics_df)
    threshold_path = out_dir / f"{name}_threshold_summary.csv"
    threshold_df.to_csv(threshold_path, index=False)
    log.info(f"Wrote {threshold_path}")

    log.info("\n--- RQ2: IQM <-> performance correlation ---")
    corr_df = compute_iqm_correlations(df)
    corr_path = out_dir / f"{name}_iqm_correlations.csv"
    corr_df.to_csv(corr_path, index=False)
    log.info(f"Wrote {corr_path}")

    log.info("\n--- Plots ---")
    plot_vs_severity(metrics_df, baseline_auc, "auc", "AUC",
                     f"{name}: AUC vs. degradation severity", out_dir / f"{name}_auc_vs_severity.png")
    plot_vs_severity(metrics_df, baseline_auc, "sensitivity", "Sensitivity (fixed threshold)",
                     f"{name}: Sensitivity vs. degradation severity", out_dir / f"{name}_sensitivity_vs_severity.png")
    plot_iqm_correlation(corr_df, out_dir / f"{name}_iqm_correlation.png")
    log.info(f"Wrote plots to {out_dir}")

    log.info("\nDone.")


if __name__ == "__main__":
    main()
