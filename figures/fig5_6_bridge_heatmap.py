#!/usr/bin/env python3
"""FIG 5 — IQM→performance bridge (AUC vs mean SSIM/Tenengrad/NoiseVar per condition).
FIG 6 — global IQM vs |Δlogit| Spearman heatmap (+ appendix scatter grid)."""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

import pubcommon as pc

M = pd.read_csv(pc.TABLES / "condition_metrics_full.csv")




def fig5():
    """One point per (deg×severity) condition; lines trace 0→6 within each deg."""
    xs = [("mean_ssim", "Mean SSIM"), ("mean_tenengrad", "Mean Tenengrad"),
          ("mean_noise_var", "Mean noise variance"), ("mean_tau", "Mean τ")]
    fig, axes = plt.subplots(2, 2, figsize=(pc.FIGW, 5.0)); axes = axes.ravel()
    clean = M[M.degradation_type == "clean"].iloc[0]
    for ax, (xc, xlab) in zip(axes, xs):
        # pooled spearman across all degraded conditions (exclude clean duplicate)
        pooled = M[M.degradation_type != "clean"]
        rho, pv = spearmanr(pooled[xc], pooled["auc"])
        for d in pc.DEG_ORDER:
            sub = M[M.degradation_type == d].sort_values("severity")
            x = [clean[xc]] + list(sub[xc]); y = [clean.auc] + list(sub.auc)
            ax.plot(x, y, "-", color=pc.DEG_COLOR[d], alpha=0.6, lw=1.3)
            ax.scatter(sub[xc], sub.auc, color=pc.DEG_COLOR[d], marker=pc.DEG_MARKER[d],
                       s=20, label=pc.DEG_LABEL[d], zorder=4)
        ax.scatter([clean[xc]], [clean.auc], color="k", marker="*", s=90, zorder=6, label="clean")
        ax.set_xlabel(xlab); ax.set_ylabel("ROC-AUC")
        ax.set_title(f"ρ(pooled) = {rho:+.2f} (p={pv:.1e})")
    # dx=-0.14 rather than the -0.10 default: these titles are nearly as wide as
    # the panel, so at the default the letter sat flush against the title's left
    # edge. Mathtext p-values ("4.3x10^-7") were tried and rejected: the exponent
    # renders at 7 pt, below the 9.5 pt floor used everywhere else, and the wider
    # title ran off the canvas.
    for ax, L in zip(axes, "ABCD"):
        pc.panel_label(ax, L, dx=-0.14)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.004), handlelength=1.6, columnspacing=1.3)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    pc.save_fig(fig, "FIG_iqm_performance_bridge_auc")

    # appendix: same vs AUPRC
    fig2, axes = plt.subplots(2, 2, figsize=(pc.FIGW, 5.0)); axes = axes.ravel()
    for ax, (xc, xlab) in zip(axes, xs):
        pooled = M[M.degradation_type != "clean"]
        rho, pv = spearmanr(pooled[xc], pooled["ap"])
        for d in pc.DEG_ORDER:
            sub = M[M.degradation_type == d].sort_values("severity")
            x = [clean[xc]] + list(sub[xc]); y = [clean.ap] + list(sub.ap)
            ax.plot(x, y, "-", color=pc.DEG_COLOR[d], alpha=0.6, lw=1.3)
            ax.scatter(sub[xc], sub.ap, color=pc.DEG_COLOR[d], marker=pc.DEG_MARKER[d], s=20, label=pc.DEG_LABEL[d])
        ax.scatter([clean[xc]], [clean.ap], color="k", marker="*", s=90, label="clean")
        ax.set_xlabel(xlab); ax.set_ylabel("AUPRC"); ax.set_title(f"ρ(pooled) = {rho:+.2f} (p={pv:.1e})", fontsize=9.5)
    for ax, L in zip(axes, "ABCD"):
        pc.panel_label(ax, L)
    axes[0].legend(loc="lower right", fontsize=7.5)
    fig2.tight_layout()
    pc.save_fig(fig2, "APP_iqm_performance_bridge_auprc", appendix=True)

    M.to_csv(pc.TABLES / "bridge_condition_table.csv", index=False)
    print("  wrote tables/bridge_condition_table.csv")


def fig6():
    R = pc.long_with_clean(pc.load_results())
    iqms = [("ssim", "SSIM"), ("tenengrad", "Tenengrad"),
            ("noise_var", "Noise variance"), ("tau", "τ"),
            ("contrast_iqr", "Contrast IQR")]
    rho_m = np.full((len(pc.DEG_ORDER), len(iqms)), np.nan)
    p_m = np.full_like(rho_m, np.nan)
    rows = []
    for i, d in enumerate(pc.DEG_ORDER):
        sub = R[R.degradation_type == d]  # severities 1-6 pooled
        for j, (m, _) in enumerate(iqms):
            dd = sub[[m, "abs_delta_logit"]].dropna()
            rho, pv = spearmanr(dd[m], dd.abs_delta_logit)
            rho_m[i, j] = rho; p_m[i, j] = pv
            rows.append(dict(degradation=d, iqm=m, n=len(dd), spearman_rho=rho, p_value=pv))
    pd.DataFrame(rows).to_csv(pc.TABLES / "global_iqm_prediction_shift_correlations.csv", index=False)

    # place the heatmap grid explicitly so it is horizontally centred on the page
    # (grid centre = 0.265 + 0.47/2 = 0.5); the colourbar lives in the balanced
    # right margin, the row labels in the balanced left margin.
    fig = plt.figure(figsize=(pc.FIGW, 3.6))
    ax = fig.add_axes([0.26, 0.17, 0.48, 0.68])
    ax.grid(False)
    im = ax.imshow(rho_m, cmap="RdBu_r", vmin=-0.6, vmax=0.6, aspect="auto")
    # two-line tick labels: single-line names collide at the 6.10 in width
    ax.set_xticks(range(len(iqms)))
    ax.set_xticklabels([l.replace(" ", "\n") for _, l in iqms])
    ax.set_yticks(range(len(pc.DEG_ORDER))); ax.set_yticklabels([pc.DEG_LABEL[d] for d in pc.DEG_ORDER])
    # sizes follow thesisstyle: ticks/cell values 9.5 pt, title and colourbar
    # label 10 pt (this figure used to sit a step above at 10/11)
    ax.tick_params(labelsize=pc.ts.SMALL)
    for i in range(len(pc.DEG_ORDER)):
        for j in range(len(iqms)):
            ax.text(j, i, f"{rho_m[i, j]:+.2f}", ha="center", va="center",
                    fontsize=pc.ts.SMALL,
                    color="white" if abs(rho_m[i, j]) > 0.35 else "black")
    cax = fig.add_axes([0.77, 0.17, 0.022, 0.68])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Spearman ρ", fontsize=pc.ts.BASE); cb.ax.tick_params(labelsize=pc.ts.SMALL)
    ax.set_title("Global IQM vs |Δlogit| (all test images)", fontsize=pc.ts.BASE)
    pc.save_fig(fig, "FIG_global_iqm_prediction_shift_heatmap")
    print("  wrote tables/global_iqm_prediction_shift_correlations.csv")

    # appendix scatter grid
    fig2, axes = plt.subplots(len(pc.DEG_ORDER), len(iqms), figsize=(pc.FIGW, 6.6))
    for i, d in enumerate(pc.DEG_ORDER):
        sub = R[R.degradation_type == d]
        for j, (m, lab) in enumerate(iqms):
            ax = axes[i, j]
            dd = sub[[m, "abs_delta_logit", "severity"]].dropna()
            sc = ax.scatter(dd[m], dd.abs_delta_logit, c=dd.severity, cmap="viridis", s=5, alpha=0.4)
            # robust trend line (lowess-like: binned medians)
            try:
                q = pd.qcut(dd[m], 12, duplicates="drop")
                g = dd.groupby(q, observed=True).agg(x=(m, "median"), y=("abs_delta_logit", "median"))
                ax.plot(g.x, g.y, "r-", lw=1.6)
            except Exception:
                pass
            if i == len(pc.DEG_ORDER) - 1:
                ax.set_xlabel(lab)
            if j == 0:
                ax.set_ylabel(f"{pc.DEG_LABEL[d]}\n|Δlogit|", fontsize=pc.ts.SMALL)
    fig2.suptitle("Global IQM vs |Δlogit|: points coloured by severity, red = binned-median trend")
    cb = fig2.colorbar(sc, ax=axes.ravel().tolist(), fraction=0.012, pad=0.01); cb.set_label("severity")
    pc.save_fig(fig2, "APP_global_iqm_prediction_shift_scattergrid", appendix=True)


if __name__ == "__main__":
    fig5()
    fig6()
