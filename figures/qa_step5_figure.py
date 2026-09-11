#!/usr/bin/env python3
"""
qa_step5_figure.py — single thesis-ready composite figure (panels A-D).
A) contrast ratios vs severity (Δμ, σ_background, CNR)
B) positives-only Spearman heatmap (CNR, Δμ, ΔCNR, ΔΔμ vs signed Δlogit)
C) example overlay: lesion ROI (red) + background ring (cyan) + breast mask (green)
D) clean vs contrast s6 scatter for Δμ and CNR
Saves PNG (300 dpi) + PDF + caption.txt.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec

import qa_common as qc

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from paths import DATA_DIR  # noqa: E402

QA = DATA_DIR / "iqm_cnr_delta_mu_qa"
PLOTS = QA / "plots"
DBG = pd.read_csv(QA / "tables" / "per_image_lesion_iqm_debug.csv")
DBG["image_id"] = DBG["image_id"].astype(str)
COR = pd.read_csv(QA / "results" / "rq2_positives_only_correlations.csv")

# ---- publication style ----
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8.5,
    "axes.linewidth": 0.8, "figure.dpi": 120,
})
RED, CYAN, GREEN = "#d62728", "#17becf", "#2ca02c"
ALPHAS = {0: 1.0, 1: 0.95, 2: 0.90, 3: 0.85, 4: 0.80, 5: 0.75, 6: 0.70}
DEG_ORDER = ["noise", "blur", "contrast", "jpeg2000", "resolution"]
METRICS = ["cnr", "delta_mu", "delta_cnr", "delta_delta_mu"]
METRIC_LBL = {"cnr": "CNR", "delta_mu": "Δμ", "delta_cnr": "ΔCNR", "delta_delta_mu": "ΔΔμ"}


def panel_label(ax, letter):
    ax.text(-0.12, 1.06, letter, transform=ax.transAxes, fontsize=15, fontweight="bold",
            va="top", ha="right")


def panelA(ax):
    d = DBG[(DBG.degradation_type.isin(["clean", "contrast"])) & (DBG.label_45 == 1)].copy()
    d["sev"] = np.where(d.condition == "clean", 0, d.severity)
    base = d[d.sev == 0].set_index("image_id")
    sevs = [0, 1, 2, 3, 4, 5, 6]
    r = {m: [] for m in ["delta_mu", "std_background", "cnr"]}
    for s in sevs:
        cur = d[d.sev == s].set_index("image_id")
        common = base.index.intersection(cur.index)
        for m in r:
            ratio = (cur.loc[common, m] / base.loc[common, m]).replace([np.inf, -np.inf], np.nan).dropna()
            r[m].append(ratio.median())
    ax.plot(sevs, [ALPHAS[s] for s in sevs], "k--", lw=1.3, zorder=1, label="contrast factor α")
    ax.plot(sevs, r["delta_mu"], "o-", color="#1f77b4", lw=1.6, ms=5, label="Δμ ratio")
    ax.plot(sevs, r["std_background"], "s-", color="#ff7f0e", lw=1.6, ms=5, label="σ$_{bg}$ ratio")
    ax.plot(sevs, r["cnr"], "^-", color=GREEN, lw=1.6, ms=6, label="CNR ratio")
    ax.axhline(1.0, color="0.6", lw=0.6, zorder=0)
    ax.set_xlabel("Contrast severity (0 = clean)")
    ax.set_ylabel("Median metric ratio (degraded / clean)")
    ax.set_title("Contrast: Δμ, σ$_{bg}$ scale with α; CNR invariant")
    ax.set_ylim(0.66, 1.03); ax.legend(loc="lower left", framealpha=0.9)
    ax.grid(alpha=0.3)


def panelB(ax):
    M = np.full((len(DEG_ORDER), len(METRICS)), np.nan)
    P = np.full_like(M, np.nan)
    sig = COR[COR.response == "signed_delta_logit"]
    for i, dg in enumerate(DEG_ORDER):
        for j, mt in enumerate(METRICS):
            row = sig[(sig.degradation == dg) & (sig.metric == mt)]
            if len(row):
                M[i, j] = row.spearman_rho.iloc[0]; P[i, j] = row.p_value.iloc[0]
    im = ax.imshow(M, cmap="RdBu_r", vmin=-0.3, vmax=0.3, aspect="auto")
    ax.set_xticks(range(len(METRICS))); ax.set_xticklabels([METRIC_LBL[m] for m in METRICS])
    ax.set_yticks(range(len(DEG_ORDER))); ax.set_yticklabels(DEG_ORDER)
    for i in range(len(DEG_ORDER)):
        for j in range(len(METRICS)):
            p = P[i, j]
            star = "***" if p < 1e-3 else ("**" if p < 1e-2 else ("*" if p < 0.05 else ""))
            ax.text(j, i, f"{M[i, j]:+.2f}\n{star}", ha="center", va="center", fontsize=8,
                    color="white" if abs(M[i, j]) > 0.18 else "black")
    ax.set_title("Spearman ρ vs signed Δlogit (positives only)")
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.set_label("Spearman ρ", fontsize=9)


def choose_example():
    cl = DBG[(DBG.condition == "clean") & (DBG.label_45 == 1) & (DBG["flags"].fillna("") == "")].copy()
    cl = cl[(cl.ring_valid_fraction > 0.98)]
    med = cl.lesion_area_px.median()
    cl["d"] = (cl.lesion_area_px - med).abs()
    # mid CNR, near-median lesion, decent ring
    cl = cl[(cl.cnr > cl.cnr.quantile(0.4)) & (cl.cnr < cl.cnr.quantile(0.7))]
    return cl.sort_values("d").iloc[0]


def panelC(ax):
    ex = choose_example()
    iid, sid, lat = ex.image_id, ex.study_id, ex.laterality
    bbox = (int(ex.lesion_xmin), int(ex.lesion_ymin), int(ex.lesion_xmax), int(ex.lesion_ymax))
    canvas = qc.reconstruct_canvas(str(sid), str(iid), str(lat))
    breast, lesion_m, ring_m = qc.build_masks(canvas, bbox, [])
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    half = max(x2 - x1, y2 - y1) // 2 + 130
    h, w = canvas.shape
    xa, xb = max(0, cx - half), min(w, cx + half)
    ya, yb = max(0, cy - half), min(h, cy + half)
    ax.imshow(canvas[ya:yb, xa:xb], cmap="gray", vmin=0, vmax=1)
    ov = np.zeros((yb - ya, xb - xa, 4)); ov[ring_m[ya:yb, xa:xb]] = [*[c / 255 for c in (23, 190, 207)], 0.45]
    ax.imshow(ov)
    ax.contour(breast[ya:yb, xa:xb].astype(float), levels=[0.5], colors=GREEN, linewidths=1.0)
    ax.add_patch(Rectangle((x1 - xa, y1 - ya), x2 - x1, y2 - y1, edgecolor=RED, facecolor="none", lw=1.8))
    ax.set_title("Lesion ROI (red), background ring (cyan),\nbreast mask (green)")
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.02, 0.02, f"CNR={ex.cnr:.2f}  Δμ={ex.delta_mu:.3f}", transform=ax.transAxes,
            color="white", fontsize=8.5, va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.55, ec="none"))
    # legend proxies
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0], [0], color=RED, lw=2, label="lesion ROI"),
                       Line2D([0], [0], color=CYAN, lw=6, alpha=0.6, label="background ring"),
                       Line2D([0], [0], color=GREEN, lw=2, label="breast mask")],
              loc="upper right", framealpha=0.85, fontsize=7.5)


def panelD(ax):
    d = DBG[(DBG.degradation_type.isin(["clean", "contrast"])) & (DBG.label_45 == 1)].copy()
    d["sev"] = np.where(d.condition == "clean", 0, d.severity)
    c0 = d[d.sev == 0].set_index("image_id"); c6 = d[d.sev == 6].set_index("image_id")
    common = c0.index.intersection(c6.index)
    ax.scatter(c0.loc[common, "delta_mu"], c6.loc[common, "delta_mu"], s=12, alpha=0.55,
               color="#1f77b4", label="Δμ")
    ax.scatter(c0.loc[common, "cnr"], c6.loc[common, "cnr"], s=12, alpha=0.55,
               color=GREEN, marker="^", label="CNR")
    hi = float(np.nanpercentile(np.r_[c0.loc[common, "cnr"], c6.loc[common, "cnr"]], 99))
    lim = [0, max(1.0, hi) * 1.05]
    ax.plot(lim, lim, "r--", lw=1.2, label="y = x (invariant)")
    ax.plot(lim, [0.70 * v for v in lim], "g:", lw=1.4, label="y = 0.70 x (α at s6)")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Clean value"); ax.set_ylabel("Contrast severity 6 value")
    ax.set_title("Clean vs contrast s6: CNR on y=x, Δμ on 0.70x")
    ax.legend(loc="upper left", framealpha=0.9); ax.grid(alpha=0.3); ax.set_aspect("equal")


def main():
    fig = plt.figure(figsize=(13, 11))
    gs = GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.26,
                  left=0.07, right=0.97, top=0.93, bottom=0.13)
    axA = fig.add_subplot(gs[0, 0]); panelA(axA); panel_label(axA, "A")
    axB = fig.add_subplot(gs[0, 1]); panelB(axB); panel_label(axB, "B")
    axC = fig.add_subplot(gs[1, 0]); panelC(axC); panel_label(axC, "C")
    axD = fig.add_subplot(gs[1, 1]); panelD(axD); panel_label(axD, "D")

    fig.suptitle("Lesion-based CNR / Δμ QA and positives-only RQ2 correlations",
                 fontsize=13.5, fontweight="bold", y=0.975)

    caption = (
        "Figure X. Quality assurance of the lesion-based contrast metrics and their relationship to model "
        "robustness (ConvNeXt-Tiny, no ColorJitter; positive annotated cases, label_45=1, n=192). "
        "(A) Median ratio of Δμ, background standard deviation σ$_{bg}$ and CNR to their clean values across "
        "contrast severities; Δμ and σ$_{bg}$ fall exactly on the contrast factor α, while CNR remains ≈1. "
        "(B) Spearman correlations of CNR, Δμ, ΔCNR and ΔΔμ with the signed logit change (pooled severities 1–6; "
        "*p<0.05, **p<0.01, ***p<0.001). (C) Representative overlay of the lesion ROI (red box), the surrounding "
        "background ring (cyan), and the breast mask (green). (D) Clean versus contrast-severity-6 values: CNR lies "
        "on the identity line (invariant) whereas Δμ lies on the 0.70x line. "
        "Because the implemented contrast reduction I' = c$_B$ + α(I − c$_B$) is linear, both the lesion–background "
        "difference (Δμ) and σ$_{bg}$ scale by α, so CNR = Δμ/σ$_{bg}$ is analytically invariant; Δμ scales with α "
        "and is therefore the appropriate lesion metric for characterising contrast degradation."
    )
    fig.text(0.07, 0.015, caption, ha="left", va="bottom", fontsize=8.2, wrap=True)

    out_png = PLOTS / "FIG_cnr_deltamu_rq2_composite.png"
    out_pdf = PLOTS / "FIG_cnr_deltamu_rq2_composite.pdf"
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)
    (PLOTS / "FIG_cnr_deltamu_rq2_composite_caption.txt").write_text(caption, encoding="utf-8")
    print("wrote", out_png.name, "+ PDF + caption.txt")


if __name__ == "__main__":
    main()
