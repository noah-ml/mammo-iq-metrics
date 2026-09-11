#!/usr/bin/env python3
"""FIG 7 — score/logit distributions (clean vs severe conditions).
FIG 8 — calibration (ECE, Brier) + fixed-threshold sensitivity/specificity vs severity.
Appendix — ROC & PR curves (clean/s3/s6) and reliability diagrams."""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve

import pubcommon as pc

R = pc.load_results()
M = pd.read_csv(pc.TABLES / "condition_metrics_full.csv")
THR = pc.clean_youden_threshold(R)


def _cond(dtype, sev):
    return R[(R.degradation_type == dtype) & (R.severity == sev)]


def fig7():
    conds = [("clean", 0, "Clean"), ("jpeg2000", 6, "JPEG2000 s6"),
             ("blur", 6, "Motion blur s6"), ("contrast", 6, "Contrast s6"),
             ("resolution", 6, "Resolution s6")]
    # 2x3 rather than 1x5: at the 6.10 in text width a five-panel row leaves
    # 1.2 in per panel, too narrow for legible 10 pt axis text
    fig, axgrid = plt.subplots(2, 3, figsize=(pc.FIGW, 4.4), sharex=True, sharey=True)
    axes = axgrid.ravel()
    bins = np.linspace(R.logit.min(), R.logit.max(), 45)
    for ax, (d, s, title) in zip(axes, conds):
        sub = _cond(d, s)
        neg = sub[sub.label_45 == 0].logit; pos = sub[sub.label_45 == 1].logit
        ax.hist(neg, bins=bins, density=True, alpha=0.5, color="#4477aa", label="negative")
        ax.hist(pos, bins=bins, density=True, alpha=0.5, color="#cc6677", label="positive")
        auc = roc_auc_score(sub.label_45, sub.prob); ap = average_precision_score(sub.label_45, sub.prob)
        ax.axvline(np.log(THR / (1 - THR)), color="k", ls="--", lw=1.0)
        ax.set_title(f"{title}\nAUC={auc:.3f}  AP={ap:.3f}")
    # axes[2] is on the top row but has no panel beneath it, so sharex would
    # otherwise leave it without tick labels
    for ax in (axgrid[1, 0], axgrid[1, 1], axgrid[0, 2]):
        ax.tick_params(labelbottom=True)
        ax.set_xlabel("logit")
    for ax in axgrid[:, 0]:
        ax.set_ylabel("density")
    axes[5].axis("off")                      # only five conditions
    axes[5].legend(*axes[0].get_legend_handles_labels(), loc="center", frameon=False)
    for ax, L in zip(axes, "ABCDE"):
        pc.panel_label(ax, L, dx=-0.08, dy=1.44)
    fig.tight_layout()
    pc.save_fig(fig, "FIG_score_distributions_clean_vs_severe")


def fig8():
    # Panel A carries two families of curves: the uncalibrated ECE and the ECE
    # after the clean-validation Platt fit (Panknin batch 3, P3-10). They differ
    # by more than an order of magnitude (0.34-0.75 raw against 0.008-0.144
    # after Platt), so panel A uses a log ordinate: on a linear axis the
    # recalibrated family collapses onto the baseline and its ordering, which is
    # the informative part, becomes invisible.
    P = pd.read_csv(pc.TABLES / "calibration_temperature_scaled_table.csv")
    platt = {(r.degradation_type, r.severity): r.ece_platt for r in P.itertuples()}

    fig, axes = plt.subplots(2, 2, figsize=(pc.FIGW, 5.0))
    specs = [(axes[0, 0], "ece", "Expected calibration error", "A"),
             (axes[0, 1], "brier", "Brier score", "B"),
             (axes[1, 0], "sensitivity", f"Sensitivity\n@ clean threshold {THR:.3f}", "C"),
             (axes[1, 1], "specificity", f"Specificity\n@ clean threshold {THR:.3f}", "D")]
    clean = M[M.degradation_type == "clean"].iloc[0]
    for ax, metric, ylab, L in specs:
        for d in pc.DEG_ORDER:
            sub = M[M.degradation_type == d].sort_values("severity")
            xs = [0] + list(sub.severity); ys = [clean[metric]] + list(sub[metric])
            ax.plot(xs, ys, marker=pc.DEG_MARKER[d], color=pc.DEG_COLOR[d], label=pc.DEG_LABEL[d])
        if metric == "ece":
            for d in pc.DEG_ORDER:
                sub = M[M.degradation_type == d].sort_values("severity")
                ys = [platt[("clean", 0)]] + [platt[(d, s)] for s in sub.severity]
                ax.plot([0] + list(sub.severity), ys, ls="--", lw=1.1,
                        marker=pc.DEG_MARKER[d], ms=3.2, mfc="none",
                        color=pc.DEG_COLOR[d])
            ax.set_yscale("log")
            # Headroom above the uncalibrated family, which tops out at 0.747, so
            # the two-row style legend sits in empty space. At the old top of 2.2
            # the headroom was 17.7 % of the axis height against a legend needing
            # about 18 %, so the second row ("after Platt scaling") landed on the
            # blur/resolution/JPEG2000 curves. 4.0 leaves about 25 %.
            ax.set_ylim(0.005, 4.0)
            style = [plt.Line2D([], [], color="0.25", ls="-", lw=1.2),
                     plt.Line2D([], [], color="0.25", ls="--", lw=1.1)]
            ax.legend(style, ["uncalibrated", "after Platt scaling"],
                      loc="upper left", frameon=False, handlelength=1.6,
                      fontsize=pc.ts.SMALL, borderaxespad=0.15,
                      labelspacing=0.2, handletextpad=0.5)
        else:
            ax.axhline(clean[metric], color="0.5", ls="--", lw=1.0)
        pc.severity_axis(ax); ax.set_ylabel(ylab); pc.panel_label(ax, L)
    h, l = axes[0, 1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.004), handlelength=1.8, columnspacing=1.3)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    pc.save_fig(fig, "FIG_calibration_threshold_vs_severity")
    M.to_csv(pc.TABLES / "calibration_threshold_table.csv", index=False)
    print("  wrote tables/calibration_threshold_table.csv")

    # appendix: F1 & balanced accuracy
    fig2, axes = plt.subplots(1, 2, figsize=(pc.FIGW, 3.0))
    for ax, metric, ylab, L in [(axes[0], "f1", "F1 @ clean threshold", "A"),
                                (axes[1], "balanced_accuracy", "Balanced accuracy @ clean threshold", "B")]:
        for d in pc.DEG_ORDER:
            sub = M[M.degradation_type == d].sort_values("severity")
            xs = [0] + list(sub.severity); ys = [clean[metric]] + list(sub[metric])
            ax.plot(xs, ys, marker=pc.DEG_MARKER[d], color=pc.DEG_COLOR[d], label=pc.DEG_LABEL[d])
        pc.severity_axis(ax); ax.set_ylabel(ylab); pc.panel_label(ax, L)
    axes[0].legend()
    fig2.tight_layout()
    pc.save_fig(fig2, "APP_f1_balacc_vs_severity", appendix=True)


def appendix_roc_pr():
    sevs = [("clean", 0, "Clean", "-"), (None, 3, "Severity 3", "--"), (None, 6, "Severity 6", ":")]
    # ROC
    fig, axes = plt.subplots(2, 3, figsize=(pc.FIGW, 4.6))
    axes = axes.ravel()
    axes[5].axis("off")
    for ax, d in zip(axes, pc.DEG_ORDER):
        for dt, s, lab, ls in sevs:
            sub = _cond("clean", 0) if s == 0 else _cond(d, s)
            fpr, tpr, _ = roc_curve(sub.label_45, sub.prob)
            ax.plot(fpr, tpr, ls=ls, color=pc.DEG_COLOR[d],
                    label=f"{lab} (AUC={roc_auc_score(sub.label_45, sub.prob):.3f})")
        ax.plot([0, 1], [0, 1], color="0.7", lw=0.8)
        ax.set_title(pc.DEG_LABEL[d]); ax.set_xlabel("FPR")
        ax.legend(fontsize=7.5, loc="lower right")
    axes[0].set_ylabel("TPR")
    fig.tight_layout(); pc.save_fig(fig, "APP_roc_curves_clean_s3_s6", appendix=True)

    # PR
    fig, axes = plt.subplots(2, 3, figsize=(pc.FIGW, 4.6))
    axes = axes.ravel()
    axes[5].axis("off")
    for ax, d in zip(axes, pc.DEG_ORDER):
        for dt, s, lab, ls in sevs:
            sub = _cond("clean", 0) if s == 0 else _cond(d, s)
            prec, rec, _ = precision_recall_curve(sub.label_45, sub.prob)
            ax.plot(rec, prec, ls=ls, color=pc.DEG_COLOR[d],
                    label=f"{lab} (AP={average_precision_score(sub.label_45, sub.prob):.3f})")
        ax.axhline(pc.N_POS / pc.N_TEST, color="0.7", lw=0.8)
        ax.set_title(pc.DEG_LABEL[d]); ax.set_xlabel("Recall")
        ax.legend(fontsize=7.5, loc="upper right")
    axes[0].set_ylabel("Precision")
    fig.tight_layout(); pc.save_fig(fig, "APP_pr_curves_clean_s3_s6", appendix=True)


def appendix_reliability():
    conds = [("clean", 0, "Clean"), ("jpeg2000", 6, "JPEG2000 s6"), ("contrast", 6, "Contrast s6")]
    fig, ax = plt.subplots(figsize=(pc.FIGW * 0.62, 3.8))
    bins = np.linspace(0, 1, 11)
    for (d, s, lab), col in zip(conds, ["#000000", "#d62728", "#2ca02c"]):
        sub = _cond(d, s); p = sub.prob.values; y = sub.label_45.values
        xc, yc = [], []
        for i in range(10):
            m = (p >= bins[i]) & (p < bins[i + 1]) if i < 9 else (p >= bins[i]) & (p <= bins[i + 1])
            if m.sum() > 0:
                xc.append(p[m].mean()); yc.append(y[m].mean())
        ax.plot(xc, yc, "o-", color=col, label=f"{lab} (ECE={pc.expected_calibration_error(y, p):.3f})")
    ax.plot([0, 1], [0, 1], "k:", lw=1.0, label="perfect")
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed frequency")
    ax.set_title("Reliability diagram"); ax.legend()
    fig.tight_layout(); pc.save_fig(fig, "APP_reliability_clean_jpeg_contrast", appendix=True)


if __name__ == "__main__":
    fig7()
    fig8()
    appendix_roc_pr()
    appendix_reliability()
