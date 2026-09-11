#!/usr/bin/env python3
"""
build_complete_splits_csv.py

Creates master_splits_1024x384_complete.csv by adding the ~930 BI-RADS 3 images
that were missing from master_splits_1024x384_5positive.csv.

Result: all 20,000 VinDr-Mammo images with correct metadata for evaluate_degradations.py.
Only reads DICOM headers (stop_before_pixels=True) — fast.
"""

import pydicom
import pandas as pd
import numpy as np
from pathlib import Path

from paths import DATA_DIR, OUT_DIR
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────

VINDR_ROOT = Path(
    r"D:\Mammo\vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-"
    r"detection-and-diagnosis-in-full-field-digital-mammography-1.0.0"
)
ANNOTATIONS_CSV = VINDR_ROOT / "breast-level_annotations.csv"
DICOM_ROOT      = VINDR_ROOT / "images"

EXISTING_CSV = DATA_DIR / "master_splits_1024x384_5positive.csv"
OUTPUT_CSV   = OUT_DIR / "master_splits_1024x384_complete.csv"

TARGET_H = 1024
TARGET_W = 384


# ── Geometry (matches dicom_to_tar_shards.py) ─────────────────────────────────

def compute_geometry(orig_h: int, orig_w: int) -> dict:
    scale     = min(TARGET_H / orig_h, TARGET_W / orig_w)
    resized_h = round(orig_h * scale)
    resized_w = round(orig_w * scale)
    pad_bottom = max(0, TARGET_H - resized_h)
    pad_right  = max(0, TARGET_W - resized_w)
    return dict(resized_height=resized_h, resized_width=resized_w,
                pad_bottom=pad_bottom, pad_right=pad_right)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Load existing CSV
    df_existing = pd.read_csv(EXISTING_CSV)
    existing_ids = set(df_existing["image_id"])
    print(f"Existing CSV : {len(df_existing):,} rows")

    # Load annotations — find all BI-RADS 3 images not yet in CSV
    ann = pd.read_csv(ANNOTATIONS_CSV)
    b3  = ann[ann["breast_birads"] == "BI-RADS 3"].copy()
    b3_missing = b3[~b3["image_id"].isin(existing_ids)].reset_index(drop=True)
    print(f"BI-RADS 3 total: {len(b3)}  |  missing from CSV: {len(b3_missing)}")

    if b3_missing.empty:
        print("Nothing to add — CSV is already complete.")
        df_existing.to_csv(OUTPUT_CSV, index=False)
        return

    # Build new rows
    new_rows = []
    errors   = []

    for _, row in tqdm(b3_missing.iterrows(), total=len(b3_missing), desc="Reading DICOM headers"):
        image_id  = row["image_id"]
        study_id  = row["study_id"]
        dicom_path = DICOM_ROOT / study_id / f"{image_id}.dicom"

        if not dicom_path.exists():
            errors.append({"image_id": image_id, "error": "DICOM not found"})
            continue

        try:
            ds = pydicom.dcmread(str(dicom_path), stop_before_pixels=True)
            orig_h       = int(ds.Rows)
            orig_w       = int(ds.Columns)
            manufacturer = str(getattr(ds, "Manufacturer", "Unknown")).strip()
        except Exception as exc:
            errors.append({"image_id": image_id, "error": str(exc)})
            continue

        geom = compute_geometry(orig_h, orig_w)

        # VinDr "testing" → "test"; "training" → "train"
        # (val distinction doesn't matter: evaluate_degradations.py filters to test only)
        vindr_split = str(row.get("split", "training")).strip().lower()
        split = "test" if vindr_split == "test" else "train"

        new_rows.append({
            "image_id":       image_id,
            "study_id":       study_id,
            "split":          split,
            "label":          0,
            "manufacturer":   manufacturer,
            "density":        str(row.get("breast_density", "")).strip(),
            "laterality":     str(row.get("laterality",     "")).strip(),
            "npy_path":       "",
            "orig_height":    orig_h,
            "orig_width":     orig_w,
            **geom,
            "breast_birads":  3,
        })

    print(f"\nBuilt {len(new_rows):,} new rows  |  {len(errors)} errors")
    if errors:
        print("Failed images:")
        for e in errors:
            print(f"  {e['image_id']}: {e['error']}")

    # Combine and save
    df_new      = pd.DataFrame(new_rows, columns=df_existing.columns)
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)

    # Sanity checks
    print(f"\n{'='*50}")
    print(f"Total rows       : {len(df_combined):,}")
    print(f"\nbreast_birads distribution:")
    print(df_combined["breast_birads"].value_counts().sort_index().to_string())
    print(f"\nsplit distribution:")
    print(df_combined["split"].value_counts().to_string())
    print(f"\nlabel distribution:")
    print(df_combined["label"].value_counts().sort_index().to_string())
    print(f"{'='*50}")

    df_combined.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
