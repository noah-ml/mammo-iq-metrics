#!/usr/bin/env python3
"""ROI-definition figure: breast mask, 96x96 homogeneous tissue patch (noise
variance), and the lesion bbox + CNR background ring (1.5x-2.5x), drawn on one
clean, presentable annotated mammogram. Faithful to evaluate_degradations.py."""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap
import matplotlib.patheffects as pe
from scipy import ndimage as ndi

import pubcommon as pc
QA = pc.DATA_DIR / "iqm_cnr_delta_mu_qa"
sys.path.insert(0, str(QA))
import qa_common as qc       # noqa: E402
import degradations as deg   # noqa: E402

PATCH = 96


# ---- replicate evaluate_degradations._setup_per_image (tissue patch) ----------
def tissue_patch(img, mask, patch_size=PATCH, contour_margin_px=80):
    h, w = img.shape
    step = max(8, patch_size // 4)
    x_g = ndi.gaussian_filter(img.astype(np.float32), sigma=1.0)
    grad = np.hypot(ndi.sobel(x_g, axis=1), ndi.sobel(x_g, axis=0))
    padded = np.pad(mask.astype(bool), 1, constant_values=False)
    dist = ndi.distance_transform_edt(padded)[1:-1, 1:-1]
    for margin in sorted({contour_margin_px, contour_margin_px // 2,
                          contour_margin_px // 4, 0}, reverse=True):
        cand = dist >= margin
        if cand.sum() < patch_size * patch_size:
            continue
        best_score, best = np.inf, None
        for y1 in range(0, h - patch_size + 1, step):
            for x1 in range(0, w - patch_size + 1, step):
                y2, x2 = y1 + patch_size, x1 + patch_size
                if cand[y1:y2, x1:x2].mean() < 1.0:
                    continue
                sc = float(grad[y1:y2, x1:x2].mean())
                if sc < best_score:
                    best_score, best = sc, (x1, y1, x2, y2)
        if best is not None:
            return best
    return None


def scaled_box(box, scale, h, w):
    x1, y1, x2, y2 = box
    cx, cy, bw, bh = 0.5 * (x1 + x2), 0.5 * (y1 + y2), x2 - x1, y2 - y1
    nw, nh = max(1, round(bw * scale)), max(1, round(bh * scale))
    return (int(max(0, min(w, round(cx - nw / 2)))), int(max(0, min(h, round(cy - nh / 2)))),
            int(max(0, min(w, round(cx + nw / 2)))), int(max(0, min(h, round(cy + nh / 2)))))


def rect(ax, box, **kw):
    x1, y1, x2, y2 = box
    ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, **kw))


# Image chosen from the patch-homogeneity scan (scan_patches.py) AND a visual
# mask-quality check (preview_candidates.py), restricted to Siemens Mammomat
# acquisitions (darker, better-contrasted than Planmed/IMS after normalisation).
# Selection criteria: clean breast mask, tissue patch on homogeneous tissue and
# well separated from the lesion, moderate (non-saturated) brightness, AND the
# full 2.5x background-ring box lying well inside the breast (ring margin ~126 px
# to the skin line) so the CNR annulus does not clip the breast edge.
PICK_IMAGE_ID = "396bea28fd4be11598bc53bf5f2b7460"


def choose_image():
    d = pd.read_csv(QA / "tables" / "per_image_lesion_iqm_debug.csv")
    d["image_id"] = d["image_id"].astype(str)
    sel = d[(d.condition == "clean") & (d.image_id == PICK_IMAGE_ID)]
    if len(sel):
        return sel.iloc[0]
    # fallback: previous heuristic
    c = d[(d.condition == "clean") & (d.label_45 == 1) & (d["flags"].fillna("") == "")
          & (d.ring_valid_fraction > 0.99)].copy()
    pool = c[c.cnr.between(0.7, 2.5)]
    return pool.sort_values(["ring_valid_fraction", "cnr"], ascending=False).iloc[0]


def main():
    ex = choose_image()
    iid, sid, lat = str(ex.image_id), str(ex.study_id), str(ex.laterality)
    box = (int(ex.lesion_xmin), int(ex.lesion_ymin), int(ex.lesion_xmax), int(ex.lesion_ymax))
    print(f"  example image_id={iid}  density={ex.density}  cnr={ex.cnr:.2f}  area={int(ex.lesion_area_px)}")

    canvas = qc.reconstruct_canvas(sid, iid, lat)
    mask = deg.get_breast_mask(canvas)
    h, w = canvas.shape
    patch = tissue_patch(canvas, mask)
    inner, outer = scaled_box(box, 1.5, h, w), scaled_box(box, 2.5, h, w)

    nz_cols = np.where(canvas.any(axis=0))[0]
    xmax = int(nz_cols[-1]) + 10
    C_MASK, C_PATCH, C_LES, C_RING = "#00d0ff", "#c026ff", "#ff2d2d", "#2ca02c"

    # Panel B zoom window (computed here so its aspect can drive width_ratios).
    pad = int(0.7 * (outer[2] - outer[0]))
    zx1, zy1 = max(0, outer[0] - pad), max(0, outer[1] - pad)
    zx2, zy2 = min(w, outer[2] + pad), min(h, outer[3] + pad)

    # Give each panel a width proportional to its image's aspect ratio so both
    # mammograms fill their axes and end up EXACTLY the same height on the page.
    aspA = xmax / h
    aspB = (zx2 - zx1) / (zy2 - zy1)

    # Authored at textwidth (6.10 in) so LaTeX places it 1:1 (no downscaling,
    # so the labels keep their true ~10 pt size), but at the SAME aspect ratio
    # as the original shipped figure (627.97 x 496.76 pt -> 1.264) so the
    # on-page footprint matches the old version. dpi bumped at save time so the
    # embedded mammograms are at least as sharp as the old 780 px rasters.
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(pc.FIGW, pc.FIGW / 1.264),
                                   gridspec_kw={"width_ratios": [aspA, aspB]})

    # Panel A: whole breast with mask outline, tissue patch, lesion box
    axA.imshow(canvas[:, :xmax], cmap="gray", vmin=0, vmax=1)
    axA.contour(mask[:, :xmax], levels=[0.5], colors=C_MASK, linewidths=1.3)
    if patch:
        rect(axA, patch, edgecolor=C_PATCH, lw=2.0)
        px = patch[0]
        axA.text((patch[0] + patch[2]) / 2, patch[1] - 14, "tissue patch",
                 color=C_PATCH, fontsize=pc.ts.BASE, ha="center", va="bottom",
                 path_effects=[pe.withStroke(linewidth=0.6, foreground="black")])
    rect(axA, box, edgecolor=C_LES, lw=2.0)
    axA.text((box[0] + box[2]) / 2, box[1] - 14, "lesion",
             color=C_LES, fontsize=pc.ts.BASE, ha="center", va="bottom",
             path_effects=[pe.withStroke(linewidth=0.6, foreground="black")])
    axA.set_title("ROI overview", fontsize=pc.ts.BASE, fontweight="bold")
    axA.set_xticks([]); axA.set_yticks([])

    # Panel B: zoom on lesion showing lesion bbox + inner/outer ring boxes
    axB.imshow(canvas[zy1:zy2, zx1:zx2], cmap="gray", vmin=0, vmax=1,
               extent=[zx1, zx2, zy2, zy1])
    # shade the ring annulus (outer minus inner), masked to breast
    ring = np.zeros((h, w))
    ox = np.zeros((h, w), bool); ox[outer[1]:outer[3], outer[0]:outer[2]] = True
    ix = np.zeros((h, w), bool); ix[inner[1]:inner[3], inner[0]:inner[2]] = True
    ring_m = ox & (~ix) & mask.astype(bool)
    ring[ring_m] = 1.0
    axB.imshow(np.ma.masked_where(ring[zy1:zy2, zx1:zx2] == 0, ring[zy1:zy2, zx1:zx2]),
               extent=[zx1, zx2, zy2, zy1], cmap=ListedColormap([C_RING]), alpha=0.35, vmin=0, vmax=1)
    rect(axB, box, edgecolor=C_LES, lw=2.2)
    rect(axB, inner, edgecolor=C_RING, lw=1.6, linestyle="--")
    rect(axB, outer, edgecolor=C_RING, lw=1.6, linestyle="--")
    axB.text(outer[0], outer[3] + 4, "background ring",
             color=C_RING, fontsize=pc.ts.BASE, va="top",
             path_effects=[pe.withStroke(linewidth=0.6, foreground="black")])
    axB.set_title("Lesion and CNR background ring", fontsize=pc.ts.BASE, fontweight="bold")
    axB.set_xticks([]); axB.set_yticks([])

    # reserve a top strip for the panel letters so they are not clipped
    fig.tight_layout(rect=(0, 0, 1, 0.952))
    # Panel letters as a standalone bold 11 pt glyph, matching every other
    # multi-panel figure in the thesis. They used to be baked into the titles as
    # "(A) ROI overview" at 10 pt, the only place that convention was used.
    for ax, L in ((axA, "A"), (axB, "B")):
        bb = ax.get_position()
        fig.text(bb.x0, 0.988, L, fontsize=pc.ts.BASE + 1, fontweight="bold",
                 va="top", ha="left")
    # Save at 200 dpi so the embedded mammogram rasters are >= the old version's
    # ~160 ppi on the page (pc.save_fig would use the 120 dpi thesis default).
    fig.savefig(pc.FIG_MAIN / "FIG_roi_definition.pdf", dpi=200)
    fig.savefig(pc.FIG_MAIN / "FIG_roi_definition.png", dpi=200)
    plt.close(fig)
    print("  wrote figures_main/FIG_roi_definition.pdf/.png at 200 dpi")


if __name__ == "__main__":
    main()
