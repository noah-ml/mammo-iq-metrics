#!/usr/bin/env python3
"""Physical anchoring of the severity levels (Panknin batch 3, P3-04/P3-15).

Two of the anchors in the thesis are stated against the NATIVE detector pitch
(0.085 mm/px) although the operators act on the 1024 x 384 working canvas. The
canvas pixel is much coarser than the detector pixel, because the breast crop
(median 2709 x 978 px) is resized onto it. This script computes the actual
canvas pitch per image from the DICOM metadata and converts the motion-blur
kernel lengths into native-equivalent millimetres, which closes the open
"motion-blur native mm" item as well.

Inputs (both local):
  dicom_analysis/vindr_metadata.csv   ImagerPixelSpacing_mm per image
  iqm_full_dataset/iqms.csv           crop_height / crop_width per image

Output:
  panknin_severity_anchors_20260820/severity_anchors.txt
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import DATA_DIR, OUT_DIR, PRIMARY_RESULTS_CSV, require  # noqa: E402

OUTDIR = OUT_DIR / "panknin_severity_anchors"
CANVAS_H = 1024
CANVAS_W = 384

BLUR_PX = [3, 5, 7, 11, 15, 21]
RES_R = [0.90, 0.80, 0.70, 0.60, 0.50, 0.40]
DOSE_F = [0.95, 0.85, 0.75, 0.65, 0.50, 0.35]
JP2_CR = [10, 25, 50, 100, 250, 500]


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    q = pd.read_csv(require(DATA_DIR / "iqms.csv", "whole-dataset IQM table"),
                    low_memory=True,
                    usecols=["image_id", "crop_height", "crop_width", "pixel_spacing_mm"])
    q = q.dropna(subset=["crop_height", "pixel_spacing_mm"]).drop_duplicates("image_id")
    # dicom_to_tar_shards resizes the breast crop with
    #   scale = min(1024 / h_crop, 384 / w_crop)
    # and zero-pads, so one canvas pixel spans native_pitch / scale.
    import numpy as _np
    inv_scale = _np.maximum(q.crop_height / CANVAS_H, q.crop_width / CANVAS_W)
    q["canvas_pitch_mm"] = q.pixel_spacing_mm * inv_scale
    p = q.canvas_pitch_mm

    L = []
    add = L.append
    add("Physical anchoring of the severity scale")
    add("images with usable geometry: %d" % len(q))
    add("")
    add("detector pixel spacing (mm/px): min %.4f  median %.4f  max %.4f"
        % (q.pixel_spacing_mm.min(), q.pixel_spacing_mm.median(), q.pixel_spacing_mm.max()))
    add("  -> spread across all four manufacturers: %.1f %%"
        % (100 * (q.pixel_spacing_mm.max() / q.pixel_spacing_mm.min() - 1)))
    add("breast-crop height (px): median %.0f  IQR %.0f-%.0f"
        % (q.crop_height.median(), q.crop_height.quantile(.25), q.crop_height.quantile(.75)))
    add("EFFECTIVE CANVAS PITCH (mm/px): median %.3f  IQR %.3f-%.3f  range %.3f-%.3f"
        % (p.median(), p.quantile(.25), p.quantile(.75), p.min(), p.max()))
    add("  -> the model input is already %.1fx coarser than the detector"
        % (p.median() / q.pixel_spacing_mm.median()))
    add("")
    add("MOTION BLUR: kernel length in native-equivalent mm (L px x canvas pitch)")
    add("  %-4s %8s %8s %8s %8s" % ("L px", "p25", "median", "p75", "max"))
    for l in BLUR_PX:
        v = l * p
        add("  %-4d %8.2f %8.2f %8.2f %8.2f" % (l, v.quantile(.25), v.median(), v.quantile(.75), v.max()))
    add("  literature: Young 2017 simulated 0.2-1.0 mm, visual detection threshold 0.4 mm;")
    add("              Abdullah 2017 simulated 0.7 and 1.5 mm, both reduce detection.")
    add("")
    add("RESOLUTION: effective pitch after the r-downsample (median canvas pitch / r)")
    add("  %-6s %10s %10s" % ("r", "median mm", "vs 0.1 mm"))
    for r in RES_R:
        v = (p / r).median()
        add("  %-6.2f %10.3f %10.1fx" % (r, v, v / 0.1))
    add("  note: even r = 1.00 gives %.3f mm, already coarser than the ~0.05-0.10 mm"
        % p.median())
    add("  pitch of FFDM detectors, so no severity level sits inside the detector range.")
    add("")
    add("DOSE-MOTIVATED NOISE: notional dose fraction")
    add("  levels: " + ", ".join("%.2f" % f for f in DOSE_F))
    add("  literature: Samei 2007 and Saunders 2007 studied half and quarter clinical dose;")
    add("              half dose remains diagnostically acceptable, quarter dose degrades")
    add("              microcalcification detection. Half dose = 0.50 = S5; S6 (0.35) lies")
    add("              between the half- and quarter-dose conditions.")
    add("")
    add("JPEG 2000: compression ratio")
    add("  levels: " + ", ".join(str(c) for c in JP2_CR))
    add("  literature: Kang 2011 (FFDM, 4 readers) found 60:1 visually indistinguishable and")
    add("              potentially acceptable for primary interpretation; visually lossless")
    add("              thresholds elsewhere are 25:1-40:1. So S3 (50:1) is the last level")
    add("              below the FFDM acceptability boundary, S2 (25:1) the conservative one.")
    rep = "\n".join(L)
    (OUTDIR / "severity_anchors.txt").write_text(rep + "\n", encoding="utf-8")
    print(rep)


if __name__ == "__main__":
    main()
