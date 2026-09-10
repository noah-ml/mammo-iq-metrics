"""Resolve input and output locations for the experiment and analysis scripts.

The released data (per-image predictions, IQM tables, ROI lookups) is not stored
in this repository. It lives in the accompanying data record, and the raw images
come from VinDr-Mammo, which requires credentialed access via PhysioNet.

Point the scripts at your local copies with environment variables:

    MAMMO_DATA_DIR   directory holding the released tables   (default: ./data)
    MAMMO_OUT_DIR    directory for generated output          (default: ./results)

Most scripts also accept an explicit path on the command line, which always
takes precedence over these defaults.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = Path(os.environ.get("MAMMO_DATA_DIR", REPO_ROOT / "data"))
OUT_DIR = Path(os.environ.get("MAMMO_OUT_DIR", REPO_ROOT / "results"))

#: Per-image predictions and image quality metrics for the primary model
#: (ConvNeXt-Tiny, seed 42, no ColorJitter). One row per image x degradation x
#: severity, 124,000 rows. Source of every figure and table in the results
#: chapter for the primary model.
PRIMARY_RESULTS_CSV = DATA_DIR / "convnext_nojitter_results_MASKEDSSIM_REFTAU.csv"

#: Per-image predictions for the three architectures x three seeds.
MULTIARCH_RESULTS_GLOB = "*_degradation_results.csv"

#: Lesion bounding boxes mapped onto the 1024x384 working canvas.
LESION_ROI_LOOKUP = DATA_DIR / "lesion_roi_lookup_1024x384.csv"

#: Whole-dataset image quality metrics, including crop geometry and pixel
#: spacing, used to convert canvas pixels into millimetres.
FULL_DATASET_IQMS = DATA_DIR / "iqms.csv"


def require(path: Path, what: str) -> Path:
    """Return ``path`` if it exists, otherwise fail with a useful message."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{what} not found at {path}\n"
            f"Set MAMMO_DATA_DIR to the directory holding the released tables, "
            f"or pass an explicit path. See docs/THESIS_MAP.md."
        )
    return Path(path)
