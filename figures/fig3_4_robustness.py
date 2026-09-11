#!/usr/bin/env python3
"""FIG 3 — AUC & AUPRC vs severity (CI).  FIG 4 — ΔAUC / ΔAUPRC forest at sev6.
Reads the precomputed condition_metrics_full.csv (no recompute of inference)."""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pubcommon as pc

M = pd.read_csv(pc.TABLES / "condition_metrics_full.csv")
CLEAN = M[M.degradation_type == "clean"].iloc[0]


def fig3():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    specs = [("auc", "ROC-AUC", CLEAN.auc), ("ap", "Average precision (AUPRC)", CLEAN.ap)]
    for ax, (metric, ylab, clean_val) in zip(axes, specs):
        for d in pc.DEG_ORDER:
            xs = [0] + list(range(1, 7))
            ys, lo, hi = [], [], []
            for s in xs:
                row = M[(M.degradation_type == ("clean" if s == 0 else d)) & (M.severity == s)].iloc[0]
                ys.append(row[metric]); lo.append(row[f"{metric}_lo"]); hi.append(row[f"{metric}_hi"])
            ax.plot(xs, ys, marker=pc.DEG_MARKER[d], color=pc.DEG_COLOR[d], label=pc.DEG_LABEL[d])
            ax.fill_between(xs, lo, hi, color=pc.DEG_COLOR[d], alpha=0.13, lw=0)
        ax.axhline(clean_val, color="0.4", ls="--", lw=1.1, label="clean baseline")
        ax.axhline(clean_val - 0.05, color="0.6", ls=":", lw=1.0, label="baseline − 0.05")
        pc.severity_axis(ax); ax.set_ylabel(ylab)
    pc.panel_label(axes[0], "A"); pc.panel_label(axes[1], "B")
    axes[0].legend(loc="lower left", framealpha=0.9, fontsize=8)
    fig.tight_layout()
    pc.save_fig(fig, "FIG_auc_auprc_vs_severity")
    M.to_csv(pc.TABLES / "robustness_auc_auprc_table.csv", index=False)
    print("  wrote tables/robustness_auc_auprc_table.csv")


def fig4():
    s6 = M[M.severity == 6].copy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for ax, metric, lo, hi, clean_val, lab in [
        (axes[0], "delta_auc", "auc_lo", "auc_hi", CLEAN.auc, "ΔAUC (degraded − clean) @ sev6"),
        (axes[1], "delta_ap", "ap_lo", "ap_hi", CLEAN.ap, "ΔAUPRC (degraded − clean) @ sev6"),
    ]:
        s = s6.sort_values(metric)  # worst (most negative) at bottom
        ypos = np.arange(len(s))
        # CI on the delta: shift the absolute-metric CI by clean baseline
        base = "auc" if metric == "delta_auc" else "ap"
        err_lo = s[metric] - (s[lo] - clean_val)
        err_hi = (s[hi] - clean_val) - s[metric]
        ax.errorbar(s[metric], ypos, xerr=[err_lo, err_hi], fmt="o", color="#333",
                    ecolor="#888", capsize=3, ms=6)
        for yi, (_, r) in zip(ypos, s.iterrows()):
            ax.plot(r[metric], yi, "o", color=pc.DEG_COLOR[r.degradation_type], ms=9, zorder=5)
        ax.set_yticks(ypos); ax.set_yticklabels([pc.DEG_LABEL[d] for d in s.degradation_type])
        ax.axvline(0, color="0.3", lw=1.1)
        ax.axvline(-0.05, color="0.6", ls=":", lw=1.0)
        ax.set_xlabel(lab)
    pc.panel_label(axes[0], "A"); pc.panel_label(axes[1], "B")
    fig.tight_layout()
    pc.save_fig(fig, "FIG_delta_auc_auprc_forest")
    s6[["degradation_type", "severity", "auc", "delta_auc", "auc_lo", "auc_hi",
        "ap", "delta_ap", "ap_lo", "ap_hi"]].to_csv(pc.TABLES / "delta_performance_table.csv", index=False)
    print("  wrote tables/delta_performance_table.csv")

    # appendix: all severities small-multiples (ΔAUC)
    fig2, ax = plt.subplots(figsize=(8, 5))
    for d in pc.DEG_ORDER:
        sub = M[M.degradation_type == d].sort_values("severity")
        ax.plot(sub.severity, sub.delta_auc, marker=pc.DEG_MARKER[d], color=pc.DEG_COLOR[d], label=pc.DEG_LABEL[d])
    ax.axhline(0, color="0.3", lw=1.0); ax.axhline(-0.05, color="0.6", ls=":", lw=1.0)
    pc.severity_axis(ax); ax.set_ylabel("ΔAUC (degraded − clean)"); ax.legend(fontsize=8)
    fig2.tight_layout()
    pc.save_fig(fig2, "APP_delta_auc_all_severities", appendix=True)


if __name__ == "__main__":
    fig3()
    fig4()
