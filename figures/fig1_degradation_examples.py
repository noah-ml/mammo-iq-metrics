#!/usr/bin/env python3
"""FIG 1 — visual degradation example grid (rows = degradation, cols = clean/s1/s3/s6).
Reconstructs ONE representative mammogram canvas from DICOM (no inference) and applies
the real degradation functions. Produces a full-image and a zoomed-lesion version,
annotated with SSIM."""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity

import pubcommon as pc

# import the verified reconstruction + mask helpers from the QA module
QA = pc.DATA_DIR / "iqm_cnr_delta_mu_qa"
sys.path.insert(0, str(QA))
import qa_common as qc            # noqa: E402
import degradations as deg        # noqa: E402

DEG_SEQ = [("noise", "Dose noise"), ("blur", "Motion blur"), ("contrast", "Contrast reduction"),
           ("jpeg2000", "JPEG2000"), ("resolution", "Resolution reduction")]
SEV_COLS = [0, 1, 3, 6]


def choose_image():
    dbg = pd.read_csv(QA / "tables" / "per_image_lesion_iqm_debug.csv")
    dbg["image_id"] = dbg["image_id"].astype(str)
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


def build(canvas, mask, sigma0, crop=None, tag="full"):
    fig, axes = plt.subplots(len(DEG_SEQ), len(SEV_COLS), figsize=(11, 13))
    for i, (dtype, dlab) in enumerate(DEG_SEQ):
        for j, s in enumerate(SEV_COLS):
            ax = axes[i, j]
            img = degrade(canvas, mask, sigma0, dtype, s)
            ssim = structural_similarity(canvas, img, data_range=1.0)
            view = img if crop is None else img[crop[2]:crop[3], crop[0]:crop[1]]
            ax.imshow(view, cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title("Clean" if s == 0 else f"Severity {s}", fontsize=11, fontweight="bold")
            if j == 0:
                ax.set_ylabel(dlab, fontsize=10, fontweight="bold")
            txt = "SSIM=1.000" if s == 0 else f"SSIM={ssim:.3f}"
            ax.text(0.5, -0.06, txt, transform=ax.transAxes, ha="center", va="top", fontsize=8)
    fig.tight_layout()
    pc.save_fig(fig, f"FIG_degradation_examples_{tag}")


def main():
    ex = choose_image()
    iid, sid, lat = ex.image_id, ex.study_id, ex.laterality
    bbox = (int(ex.lesion_xmin), int(ex.lesion_ymin), int(ex.lesion_xmax), int(ex.lesion_ymax))
    print(f"  example image_id={iid}  lesion_area={int(ex.lesion_area_px)}  cnr={ex.cnr:.2f}")
    canvas = qc.reconstruct_canvas(str(sid), str(iid), str(lat))
    mask = deg.get_breast_mask(canvas)
    sigma0 = deg.get_sigma0(canvas, mask)

    # full image: trim trailing all-zero pad columns/rows for a tighter view
    nz_rows = np.where(canvas.any(axis=1))[0]; nz_cols = np.where(canvas.any(axis=0))[0]
    full_crop = (0, int(nz_cols[-1]) + 8, 0, int(nz_rows[-1]) + 8)
    build(canvas, mask, sigma0, crop=full_crop, tag="full")

    # zoomed lesion patch
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    half = max(x2 - x1, y2 - y1) // 2 + 90
    h, w = canvas.shape
    zoom = (max(0, cx - half), min(w, cx + half), max(0, cy - half), min(h, cy + half))
    build(canvas, mask, sigma0, crop=zoom, tag="zoom")


if __name__ == "__main__":
    main()
