#!/usr/bin/env python3
"""Reattach the VinDr-Mammo annotation columns to a released prediction table.

The released tables carry the model outputs and the image quality metrics, which
are outputs of this work, but not ``label_45``, ``label_5``, ``density`` or
``manufacturer``. Those are VinDr-Mammo annotations, distributed under a
credentialed PhysioNet licence, and this project is not permitted to
redistribute them.

Anyone with their own approved copy of the dataset can restore them in one step:

    python reproduce/join_labels.py \\
        --predictions data/convnext_nojitter_results_STRIPPED.csv \\
        --breast-annotations /path/to/vindr/breast-level_annotations.csv \\
        --output data/convnext_nojitter_results_MASKEDSSIM_REFTAU.csv

``label_45`` is the positive class used throughout the thesis: BI-RADS 4 or 5
against BI-RADS 1, 2 or 3. ``label_5`` is the stricter BI-RADS 5 variant, kept
for the sensitivity analysis. Both are derived here from ``breast_birads``
rather than shipped, so the definition is visible instead of implied.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

BIRADS_COL = "breast_birads"
NEEDED = ["image_id", BIRADS_COL, "breast_density", "laterality", "view_position"]


def birads_number(value) -> float:
    """'BI-RADS 4' -> 4.0. Returns NaN for anything unparseable."""
    try:
        return float(str(value).strip().rsplit(maxsplit=1)[-1])
    except (ValueError, IndexError):
        return float("nan")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predictions", required=True, type=Path)
    p.add_argument("--breast-annotations", required=True, type=Path,
                   help="breast-level_annotations.csv from your VinDr-Mammo copy")
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()

    preds = pd.read_csv(a.predictions)
    if "image_id" not in preds.columns:
        sys.exit("the prediction table has no image_id column")

    ann = pd.read_csv(a.breast_annotations)
    missing = [c for c in (BIRADS_COL,) if c not in ann.columns]
    if missing:
        sys.exit(f"{a.breast_annotations.name} has no {missing} column; "
                 f"expected the official VinDr-Mammo breast-level annotations")

    keep = [c for c in NEEDED if c in ann.columns]
    ann = ann[keep].drop_duplicates("image_id").copy()

    n = ann[BIRADS_COL].map(birads_number)
    ann["label_45"] = (n >= 4).astype("Int64")
    ann["label_5"] = (n >= 5).astype("Int64")
    if "breast_density" in ann.columns:
        ann = ann.rename(columns={"breast_density": "density"})

    out = preds.merge(ann, on="image_id", how="left", validate="many_to_one")

    unmatched = int(out["label_45"].isna().sum())
    if unmatched:
        print(f"WARNING: {unmatched} of {len(out)} rows found no annotation. "
              f"Check that the prediction table and the annotation file come "
              f"from the same dataset version.", file=sys.stderr)

    out.to_csv(a.output, index=False)
    print(f"wrote {a.output}  ({len(out):,} rows, "
          f"{int(out['label_45'].sum()):,} positive rows)")
    return 1 if unmatched else 0


if __name__ == "__main__":
    raise SystemExit(main())
