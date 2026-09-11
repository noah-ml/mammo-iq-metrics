"""Regenerate Figure 4.3 (full-image degradation example grid) — TIGHT layout, PDF.
Same data/degradations as the original fig1_degradation_examples.py, but:
 - figure size matched to the montage aspect so there is no horizontal whitespace
   between the (tall, narrow) mammogram columns;
 - near-zero wspace/hspace;
 - SSIM annotation moved INSIDE each panel (bottom strip) instead of below it, so
   rows pack tightly;
 - row labels (degradation) on the left, severity titles on top.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import pubcommon as pc
QA = pc.DATA_DIR / "iqm_cnr_delta_mu_qa"
sys.path.insert(0, str(QA))
import qa_common as qc
import degradations as deg

OUTDIR = pc.FIG_MAIN  # write straight into the publication figures_main dir
DEG_SEQ = [("noise", "Dose noise"), ("blur", "Motion blur"), ("contrast", "Contrast reduction"),
           ("jpeg2000", "JPEG2000"), ("resolution", "Resolution reduction")]
# Row labels are set horizontally in the left margin, wrapped onto two lines.
# As rotated y-labels they ran along the row HEIGHT, and in the zoom montage a
# row is only 0.74 in tall against a 1.6 in label, so the five labels overlapped
# each other and the last one was clipped by the canvas edge.
ROW_LABEL = {"Dose noise": "Dose\nnoise", "Motion blur": "Motion\nblur",
             "Contrast reduction": "Contrast\nreduction", "JPEG2000": "JPEG2000",
             "Resolution reduction": "Resolution\nreduction"}
# PPT-style: clean reference plus all six severity levels.
SEV_COLS = [0, 1, 2, 3, 4, 5, 6]

# Siemens CC-view breast (widest, most CC-like half-dome among the screened
# candidates; scan_cc_views.py), density C with clear parenchymal texture so the
# degradation detail loss is visible, and carrying the annotated lesion so the
# lesion zoom (Fig. 4.4) stays the same case. Chosen to mirror the presentation
# montages.
PICK_IMAGE_ID = "3b46267ffa238158eda9aa2a63a2c5ed"


def choose_image():
    dbg = pd.read_csv(QA / "tables" / "per_image_lesion_iqm_debug.csv")
    dbg["image_id"] = dbg["image_id"].astype(str)
    sel = dbg[(dbg.condition == "clean") & (dbg.image_id == PICK_IMAGE_ID)]
    if len(sel):
        return sel.iloc[0]
    cl = dbg[(dbg.condition == "clean") & (dbg.label_45 == 1) & (dbg["flags"].fillna("") == "")].copy()
    cl = cl[(cl.ring_valid_fraction > 0.98)]
    lo, hi = cl.lesion_area_px.quantile(0.55), cl.lesion_area_px.quantile(0.85)
    cl = cl[(cl.lesion_area_px >= lo) & (cl.lesion_area_px <= hi)]
    return cl.sort_values("cnr", ascending=False).iloc[0]


def degrade(canvas, mask, sigma0, dtype, sev):
    if sev == 0:
        return canvas
    return deg.apply_degradation(canvas, dtype, sev, mask=mask, sigma0=sigma0,
                                 pixel_spacing_mm=0.07, seed=2026)


def build(canvas, mask, sigma0, crop, tag):
    c0, c1, r0, r1 = crop
    Hc, Wc = (r1 - r0), (c1 - c0)
    asp = Wc / Hc  # image width/height
    print(f"  [{tag}] cropped {Wc}x{Hc}px  (w/h aspect={asp:.3f})")

    nrow, ncol = len(DEG_SEQ), len(SEV_COLS)
    # Size the figure to the box LaTeX will place it in, so that it is drawn
    # 1:1 and the in-figure text keeps its point size (jour fixe 2026-07-29).
    # Both are now included at width=\textwidth: at 1.02 in of row-label margin
    # "full" comes out 6.10 x 7.89 in, which fits a float page, so the height cap
    # below never binds and the montage is placed 1:1 in both cases. ("full" used
    # to be included at height=0.86\textheight, which would have scaled a 6.10 in
    # canvas up past the text width.)
    TARGET_W = 6.10                                  # \textwidth = 15.5 cm
    TARGET_H = {"full": 8.25}.get(tag)               # safety cap, inches
    left_lab, top_lab = 1.02, 0.42   # margins for row labels / column titles
    # each cell exactly matches the image aspect -> no horizontal gaps
    cell_w = (TARGET_W - left_lab) / ncol
    cell_h = cell_w / asp
    if TARGET_H is not None and nrow * cell_h + top_lab > TARGET_H:
        cell_h = (TARGET_H - top_lab) / nrow         # height is the binding limit
        cell_w = cell_h * asp
    fig_w = ncol * cell_w + left_lab
    fig_h = nrow * cell_h + top_lab
    print(f"  [{tag}] figure {fig_w:.2f} x {fig_h:.2f} in (drawn 1:1)")
    fig = plt.figure(figsize=(fig_w, fig_h))

    gs = fig.add_gridspec(nrow, ncol, wspace=0.015, hspace=0.015,
                          left=left_lab / fig_w, right=0.999,
                          bottom=0.002, top=1 - top_lab / fig_h)

    for i, (dtype, dlab) in enumerate(DEG_SEQ):
        for j, s in enumerate(SEV_COLS):
            ax = fig.add_subplot(gs[i, j])
            img = degrade(canvas, mask, sigma0, dtype, s)
            # mask-averaged SSIM, matching the deployed pipeline (recompute_ssim_tau.masked_ssim)
            _, _smap = structural_similarity(canvas.astype(np.float32), img.astype(np.float32),
                                             data_range=1.0, full=True)
            _m = mask.astype(bool)
            ssim = float(_smap[_m].mean()) if _m.any() else float(_smap.mean())
            view = img[r0:r1, c0:c1]
            ax.imshow(view, cmap="gray", vmin=0, vmax=1, aspect="equal")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color("0.6"); sp.set_linewidth(0.4)
            if i == 0:
                ax.set_title("Clean" if s == 0 else f"S{s}",
                             fontsize=pc.ts.BASE, fontweight="bold", pad=4)
            txt = "1.000" if s == 0 else f"{ssim:.3f}"
            ax.text(0.5, 0.030, txt, transform=ax.transAxes, ha="center", va="bottom",
                    fontsize=pc.ts.BASE, color="white",  # matches the SSIM annotations in Fig 4.2
                    bbox=dict(boxstyle="round,pad=0.14", fc="black", ec="none", alpha=0.6))

    # horizontal row labels, vertically centred on each row of the gridspec
    gs_top, gs_bot, gs_left = 1 - top_lab / fig_h, 0.002, left_lab / fig_w
    for i, (dtype, dlab) in enumerate(DEG_SEQ):
        yc = gs_top - (i + 0.5) * (gs_top - gs_bot) / nrow
        fig.text(gs_left - 0.010, yc, ROW_LABEL[dlab], ha="right", va="center",
                 multialignment="right", fontsize=pc.ts.SMALL, fontweight="bold")

    for ext in ("pdf", "png"):
        fig.savefig(OUTDIR / f"FIG_degradation_examples_{tag}.{ext}",
                    dpi=300 if ext == "png" else None, )
    plt.close(fig)
    print(f"  wrote FIG_degradation_examples_{tag}.pdf/.png -> {OUTDIR}")


def main():
    ex = choose_image()
    iid, sid, lat = ex.image_id, ex.study_id, ex.laterality
    print(f"  example image_id={iid} lesion_area={int(ex.lesion_area_px)} cnr={ex.cnr:.2f}")
    canvas = qc.reconstruct_canvas(str(sid), str(iid), str(lat))
    mask = deg.get_breast_mask(canvas)
    sigma0 = deg.get_sigma0(canvas, mask)
    h, w = canvas.shape

    # full breast: drop trailing all-zero pad rows/cols
    nz_rows = np.where(canvas.any(axis=1))[0]
    nz_cols = np.where(canvas.any(axis=0))[0]
    full_crop = (int(nz_cols[0]), int(nz_cols[-1]) + 1,
                 int(nz_rows[0]), int(nz_rows[-1]) + 1)
    build(canvas, mask, sigma0, full_crop, "full")

    # zoom: square window centred on the annotated lesion
    x1, y1, x2, y2 = (int(ex.lesion_xmin), int(ex.lesion_ymin),
                      int(ex.lesion_xmax), int(ex.lesion_ymax))
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    half = max(x2 - x1, y2 - y1) // 2 + 90
    zoom_crop = (max(0, cx - half), min(w, cx + half),
                 max(0, cy - half), min(h, cy + half))
    build(canvas, mask, sigma0, zoom_crop, "zoom")


if __name__ == "__main__":
    main()
