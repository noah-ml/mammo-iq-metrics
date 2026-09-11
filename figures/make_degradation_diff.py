#!/usr/bin/env python3
"""Degradation difference-heatmap montage (candidate replacement for the
main-text photographic montage; §4.3). Two versions for the author to pick:

  signed : (degraded - clean), diverging colormap, symmetric shared scale.
  abs    : |degraded - clean|, sequential colormap, shared scale.

Grid = 5 degradation types (rows) x severities 1..6 (cols). The clean column
is omitted (its difference is identically zero). Differences are masked to the
breast region and shown on the lesion-zoom crop, matching Fig 4.4. A single
shared colourbar per figure gives the Farbskala requested at the jour fixe.

Reuses the exact pipeline of fig_degradation_examples_tight.py
(qa_common.reconstruct_canvas + degradations.apply_degradation), same image
(3b46267f...), same seed. No model inference involved.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from skimage.metrics import structural_similarity

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import pubcommon as pc  # applies shared rcParams (DejaVu Sans, B8-consistent)
QA = DATA_DIR / "iqm_cnr_delta_mu_qa"   # QA tables from the data record
sys.path.insert(0, str(QA))
import qa_common as qc
import degradations as deg

HERE = Path(__file__).resolve().parent
import sys as _sys
_sys.path.insert(0, str(HERE.parent / "experiments"))
from paths import DATA_DIR, OUT_DIR  # noqa: E402
# two-line row labels: at the 1:1 figure size a row is ~1.1 in tall and the
# single-line names run past it
DEG_SEQ = [("noise", "Dose-motivated\nnoise"), ("blur", "Motion blur"),
           ("contrast", "Contrast\nreduction"), ("jpeg2000", "JPEG2000\ncompression"),
           ("resolution", "Resolution\nreduction")]
SEV_COLS = [1, 2, 3, 4, 5, 6]
PICK_IMAGE_ID = "3b46267ffa238158eda9aa2a63a2c5ed"


def choose_image():
    dbg = pd.read_csv(QA / "tables" / "per_image_lesion_iqm_debug.csv")
    dbg["image_id"] = dbg["image_id"].astype(str)
    sel = dbg[(dbg.condition == "clean") & (dbg.image_id == PICK_IMAGE_ID)]
    return sel.iloc[0]


def load():
    ex = choose_image()
    canvas = qc.reconstruct_canvas(str(ex.study_id), str(ex.image_id), str(ex.laterality))
    mask = deg.get_breast_mask(canvas)
    sigma0 = deg.get_sigma0(canvas, mask)
    h, w = canvas.shape
    x1, y1, x2, y2 = (int(ex.lesion_xmin), int(ex.lesion_ymin),
                      int(ex.lesion_xmax), int(ex.lesion_ymax))
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    half = max(x2 - x1, y2 - y1) // 2 + 90
    crop = (max(0, cx - half), min(w, cx + half), max(0, cy - half), min(h, cy + half))
    return canvas, mask, sigma0, crop, ex


def compute_diffs(canvas, mask, sigma0, crop):
    """Return dict[(dtype,sev)] -> (masked diff crop, ssim). Also global vmax99."""
    c0, c1, r0, r1 = crop
    mcrop = mask[r0:r1, c0:c1].astype(bool)
    diffs, ssims, allabs = {}, {}, []
    for dtype, _ in DEG_SEQ:
        for s in SEV_COLS:
            dg = deg.apply_degradation(canvas, dtype, s, mask=mask, sigma0=sigma0,
                                       pixel_spacing_mm=0.07, seed=2026)
            _, smap = structural_similarity(canvas.astype(np.float32), dg.astype(np.float32),
                                            data_range=1.0, full=True)
            ssims[(dtype, s)] = float(smap[mask.astype(bool)].mean())
            d = (dg - canvas)[r0:r1, c0:c1]
            dm = np.where(mcrop, d, np.nan)
            diffs[(dtype, s)] = dm
            allabs.append(np.abs(dm[mcrop]))
    vmax = float(np.nanpercentile(np.concatenate(allabs), 99.0))
    return diffs, ssims, vmax, mcrop


def render(diffs, ssims, vmax, signed: bool, ex):
    nrow, ncol = len(DEG_SEQ), len(SEV_COLS)
    cmap = plt.get_cmap("RdBu_r" if signed else "magma").copy()
    cmap.set_bad("white")
    norm = plt.Normalize(-vmax, vmax) if signed else plt.Normalize(0.0, vmax)

    # Drawn 1:1 at the \textwidth of 15.5 cm so the in-figure text keeps its
    # point size on the page (jour fixe 2026-07-29). The colourbar sits along
    # the BOTTOM (horizontal) rather than the right, so the image grid keeps the
    # full text width and the panels stay their original on-page size while the
    # legend fonts can be enlarged without stealing width from the grid.
    fig_w = pc.FIGW
    LEFT, RIGHT, TOP, BOTTOM = 0.085, 0.995, 0.965, 0.185
    WSPACE = HSPACE = 0.03
    # size fig_h so each square panel tiles its axes box with no letterboxing:
    Hpx, Wpx = next(iter(diffs.values())).shape          # panel pixel aspect
    img_aspect = Wpx / Hpx
    cell_w_in = (RIGHT - LEFT) * fig_w / (ncol + (ncol - 1) * WSPACE)
    cell_h_in = cell_w_in / img_aspect
    grid_h_in = cell_h_in * (nrow + (nrow - 1) * HSPACE)
    fig_h = grid_h_in / (TOP - BOTTOM)
    fig, axes = plt.subplots(nrow, ncol, figsize=(fig_w, fig_h),
                             gridspec_kw=dict(wspace=WSPACE, hspace=HSPACE))
    for i, (dtype, dlab) in enumerate(DEG_SEQ):
        for j, s in enumerate(SEV_COLS):
            ax = axes[i, j]
            d = diffs[(dtype, s)]
            ax.imshow(d if signed else np.abs(d), cmap=cmap, norm=norm, aspect="equal",
                      interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color("0.6"); sp.set_linewidth(0.4)
            if i == 0:
                ax.set_title(f"S{s}", fontsize=pc.ts.BASE, fontweight="bold", pad=4)
            if j == 0:
                ax.set_ylabel(dlab, fontsize=pc.ts.SMALL, fontweight="bold")
            ax.text(0.5, 0.012, f"{ssims[(dtype, s)]:.3f}", transform=ax.transAxes,
                    ha="center", va="bottom", fontsize=pc.ts.BASE,
                    color="black" if not signed else "0.15",
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.65))

    fig.subplots_adjust(left=LEFT, right=RIGHT, top=TOP, bottom=BOTTOM)
    # horizontal colourbar centred under the grid, in the reserved bottom margin
    cax = fig.add_axes([0.30, 0.105, 0.42, 0.018])
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    # Colourbar follows the thesis-wide role sizes: ticks at SMALL (9.5 pt) like
    # every other tick label, label at BASE (10 pt) like every other axis label.
    # These were at BASE+1 (11 pt), which is the panel-letter size and made this
    # the only colourbar in the thesis larger than the body figures' axis text
    # (the Fig 5.5 heatmap colourbar now matches at 9.5/10).
    cb.ax.tick_params(labelsize=pc.ts.SMALL)
    cb.set_label("intensity change (degraded $-$ clean), normalized units"
                 if signed else "|intensity change| (degraded $-$ clean), normalized units",
                 fontsize=pc.ts.BASE)
    tag = "signed" if signed else "abs"
    fig.savefig(HERE / f"FIG_degradation_diff_{tag}.pdf", dpi=200)
    fig.savefig(HERE / f"FIG_degradation_diff_{tag}.png", dpi=200)
    plt.close(fig)
    print(f"wrote FIG_degradation_diff_{tag}.pdf/.png")


def main():
    canvas, mask, sigma0, crop, ex = load()
    print(f"image={ex.image_id} lesion_area={int(ex.lesion_area_px)} cnr={ex.cnr:.2f}")
    diffs, ssims, vmax, _ = compute_diffs(canvas, mask, sigma0, crop)
    print(f"shared |diff| 99th-pct scale = {vmax:.4f}")
    render(diffs, ssims, vmax, signed=True, ex=ex)
    render(diffs, ssims, vmax, signed=False, ex=ex)


if __name__ == "__main__":
    main()
