#!/usr/bin/env python3
"""
build_lesion_roi_lookup.py
==========================
Builds a small lookup table mapping each annotated test-split lesion bbox
(from finding_annotations.csv, given in ORIGINAL DICOM pixel coordinates)
to its location in the final 1024x384 preprocessed canvas.

Why this exists
---------------
evaluate_degradations.py runs on the cluster against TAR-shard .npy arrays
only — it has no access to the raw DICOMs, so it cannot recompute lesion
bbox locations on the fly. This script does that work once, locally (where
the raw VinDr-Mammo DICOMs are available), by replicating the exact
geometric steps of dicom_to_tar_shards.py::_process_one:

    1. RescaleSlope/Intercept + MONOCHROME1 inversion   (no spatial change)
    2. Percentile [0.5, 99.5] normalisation             (no spatial change)
    3. Horizontal flip for laterality == "R"            (x' = W_orig - x)
    4. Burned-in text-annotation removal                (zeroes pixels only)
    5. Tight crop to nonzero bbox + crop_margin_px=10   (subtract crop origin)
    6. Resize with scale = min(target_h/h_crop, target_w/w_crop)
    7. Zero-pad to (1024, 384), breast anchored top-left (no further shift)

The resulting CSV is small (~357 rows) and travels with the splits CSV to
the cluster; evaluate_degradations.py just looks up bboxes by image_id —
no DICOM access needed at evaluation time.

Output columns
--------------
    image_id, study_id, laterality,
    lesion_xmin, lesion_ymin, lesion_xmax, lesion_ymax   (primary lesion, canvas coords)
    n_findings, other_lesion_bboxes                       (JSON list of [x1,y1,x2,y2])

Usage
-----
    python build_lesion_roi_lookup.py \
        --annotations <VinDr root>/finding_annotations.csv \
        --dicom-dir   <VinDr root>/images \
        --output      lesion_roi_lookup_1024x384.csv
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from scipy.ndimage import label as ndimage_label

warnings.filterwarnings("ignore")

TARGET_H = 1024
TARGET_W = 384
CROP_MARGIN_PX = 10  # must match dicom_to_tar_shards.py default (--crop-margin-px)

BBox = tuple[float, float, float, float]


# ---------------------------------------------------------------------------
# Geometry replication (mirrors dicom_to_tar_shards.py::_process_one steps 1-6)
# ---------------------------------------------------------------------------

def _remove_text_annotations(arr: np.ndarray) -> np.ndarray:
    """Zero all connected foreground components except the largest (the breast).

    Byte-for-byte copy of dicom_to_tar_shards.py::_remove_text_annotations —
    must match exactly so the resulting crop bbox is identical.
    """
    mask = arr > 0
    labeled, n_comp = ndimage_label(mask)
    if n_comp <= 1:
        return arr
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    largest = int(sizes.argmax())
    clean = arr.copy()
    clean[labeled != largest] = 0.0
    return clean


def _derive_geometry(dicom_path: Path, laterality: str) -> tuple[int, tuple[int, int, int, int], float]:
    """Replicate _process_one steps 1-6 and return (W_orig, crop_bbox, scale).

    crop_bbox = (ymin, ymax, xmin, xmax) in POST-FLIP image coordinates —
    i.e. the same crop window dicom_to_tar_shards.py applied right before
    the resize step.
    """
    ds = pydicom.dcmread(str(dicom_path))
    image = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    inter = float(getattr(ds, "RescaleIntercept", 0.0))
    image = image * slope + inter
    if getattr(ds, "PhotometricInterpretation", "").strip().upper() == "MONOCHROME1":
        image = np.max(image) - image

    H_orig, W_orig = image.shape

    p_low = float(np.percentile(image, 0.5))
    p_high = float(np.percentile(image, 99.5))
    if p_high > p_low:
        image = np.clip(image, p_low, p_high)
        image = (image - p_low) / (p_high - p_low)
    else:
        image = np.zeros_like(image)

    if laterality == "R":
        image = np.ascontiguousarray(np.flip(image, axis=1))

    image = _remove_text_annotations(image)

    rows = np.any(image > 0, axis=1)
    cols = np.any(image > 0, axis=0)
    if not rows.any():
        raise ValueError("all-zero image after text-annotation removal")
    H, W = image.shape
    ymin = max(0, int(np.where(rows)[0][0]) - CROP_MARGIN_PX)
    ymax = min(H, int(np.where(rows)[0][-1]) + 1 + CROP_MARGIN_PX)
    xmin = max(0, int(np.where(cols)[0][0]) - CROP_MARGIN_PX)
    xmax = min(W, int(np.where(cols)[0][-1]) + 1 + CROP_MARGIN_PX)

    h_crop, w_crop = ymax - ymin, xmax - xmin
    scale = min(TARGET_H / h_crop, TARGET_W / w_crop)

    return W_orig, (ymin, ymax, xmin, xmax), scale


def _transform_bbox(
    bbox: BBox,
    W_orig: int,
    laterality: str,
    crop: tuple[int, int, int, int],
    scale: float,
) -> tuple[int, int, int, int]:
    """Map a lesion bbox from original DICOM coords to final 1024x384 canvas coords."""
    x1, y1, x2, y2 = bbox

    # Step 3: laterality flip (x' = W_orig - x), y unchanged
    if laterality == "R":
        x1, x2 = W_orig - x2, W_orig - x1

    # Step 5: crop shift
    ymin_c, ymax_c, xmin_c, xmax_c = crop
    x1, x2 = x1 - xmin_c, x2 - xmin_c
    y1, y2 = y1 - ymin_c, y2 - ymin_c

    # Step 6: uniform resize scale
    x1, x2, y1, y2 = x1 * scale, x2 * scale, y1 * scale, y2 * scale

    # Step 7: pad — top-left anchored, no further shift

    x1 = int(np.clip(round(x1), 0, TARGET_W))
    x2 = int(np.clip(round(x2), 0, TARGET_W))
    y1 = int(np.clip(round(y1), 0, TARGET_H))
    y2 = int(np.clip(round(y2), 0, TARGET_H))
    return x1, y1, x2, y2


def _bbox_area(b: BBox) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--annotations", type=str, required=True, help="finding_annotations.csv")
    p.add_argument("--dicom-dir",   type=str, required=True, help="VinDr images/ root (contains <study_id>/<image_id>.dicom)")
    p.add_argument("--output",      type=str, default="lesion_roi_lookup_1024x384.csv")
    p.add_argument("--split",       type=str, default="test")
    args = p.parse_args()

    ann = pd.read_csv(args.annotations)
    ann.loc[ann["split"] == "testing", "split"] = "test"
    ann["image_id"] = ann["image_id"].astype(str)
    ann["study_id"] = ann["study_id"].astype(str)

    sub = ann[
        (ann["split"] == args.split)
        & ann["xmin"].notna()
        & (ann["finding_categories"] != "['No Finding']")
    ].copy()

    dicom_dir = Path(args.dicom_dir)
    rows_out: list[dict] = []
    n_ok, n_fail = 0, 0

    for image_id, grp in sub.groupby("image_id"):
        study_id   = str(grp.iloc[0]["study_id"])
        laterality = str(grp.iloc[0]["laterality"]).strip().upper()
        dicom_path = dicom_dir / study_id / f"{image_id}.dicom"
        if not dicom_path.exists():
            print(f"  [skip] {image_id}: DICOM not found at {dicom_path}")
            n_fail += 1
            continue

        bboxes: list[BBox] = [
            (float(r["xmin"]), float(r["ymin"]), float(r["xmax"]), float(r["ymax"]))
            for _, r in grp.iterrows()
        ]

        try:
            W_orig, crop, scale = _derive_geometry(dicom_path, laterality)
            canvas_bboxes = [_transform_bbox(b, W_orig, laterality, crop, scale) for b in bboxes]
        except Exception as exc:
            print(f"  [fail] {image_id}: {exc}")
            n_fail += 1
            continue

        # Primary = largest-area bbox (mirrors roi_metrics_dicom_3.0.select_primary_lesion_bbox)
        areas = [_bbox_area(b) for b in canvas_bboxes]
        primary_idx = int(np.argmax(areas))
        primary = canvas_bboxes[primary_idx]
        others  = [b for i, b in enumerate(canvas_bboxes) if i != primary_idx]

        rows_out.append({
            "image_id":    image_id,
            "study_id":    study_id,
            "laterality":  laterality,
            "lesion_xmin": primary[0],
            "lesion_ymin": primary[1],
            "lesion_xmax": primary[2],
            "lesion_ymax": primary[3],
            "n_findings":  len(canvas_bboxes),
            "other_lesion_bboxes": json.dumps(others),
        })
        n_ok += 1

    out_df = pd.DataFrame(rows_out)
    out_df.to_csv(args.output, index=False)
    print(f"\nWrote {len(out_df)} rows to {args.output}  (ok={n_ok}, failed/skipped={n_fail})")
    if len(out_df):
        widths  = out_df["lesion_xmax"] - out_df["lesion_xmin"]
        heights = out_df["lesion_ymax"] - out_df["lesion_ymin"]
        print(f"Lesion bbox size in canvas: width  median={widths.median():.1f}  range=[{widths.min():.0f}, {widths.max():.0f}]")
        print(f"                            height median={heights.median():.1f}  range=[{heights.min():.0f}, {heights.max():.0f}]")
        n_degenerate = int(((widths <= 0) | (heights <= 0)).sum())
        print(f"Degenerate bboxes (zero-area after clipping): {n_degenerate}")


if __name__ == "__main__":
    main()
