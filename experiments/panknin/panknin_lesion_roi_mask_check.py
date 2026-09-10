#!/usr/bin/env python3
"""Why the contrast invariance is exact for 169 of 192 lesions and not for 23.

Origin
------
While fixing the lesion-severity claim (Panknin batch 3, P3-09) the contrast
sweep was checked against its analytic prediction: under
I' = c_B + alpha (I - c_B) applied to in-mask pixels, both ROI means shift the
same way, so delta_mu should scale exactly by alpha and the CNR should be
exactly invariant. It does for most images but not all.

Mechanism under test
--------------------
In `evaluate_degradations.lesion_background_ring`:
    ring   = outer_box & ~inner_box & breast_mask     <- intersected with mask
    lesion = bbox_mask(lesion_bbox)                   <- NOT intersected
So a lesion box extending beyond the breast mask mixes scaled in-mask pixels
with UNSCALED out-of-mask pixels while the ring stays purely in-mask. With
in-mask fraction f and out-of-mask value v,
    mean_lesion'(alpha) = f (c + alpha (m_in - c)) + (1 - f) v
    mean_ring'(alpha)   = c + alpha (m_ring - c)
so delta_mu'(alpha) is AFFINE in alpha with an intercept that vanishes iff
f = 1. That is testable from the results CSV alone, with no image loading.

Result (2026-08-20): 23 of 192 lesions deviate. For the 169 conforming ones the
fitted slope equals the clean delta_mu to 5e-6 and the intercept is ~1e-5
relative; the 23 have a large intercept (median 0.76 relative). The deviating
boxes are much larger (median 203 x 237 px against 85 x 88 px on the
1024 x 384 canvas), which is the expected signature. Excluding them changes no
reported lesion coefficient by more than 0.04 and leaves the median ratios
unchanged, so nothing reported depends on it.

Output
------
  panknin_lesion_roi_check_20260820/roi_mask_check.csv
  panknin_lesion_roi_check_20260820/roi_mask_report.txt
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import DATA_DIR, OUT_DIR, PRIMARY_RESULTS_CSV, require  # noqa: E402

RESULTS_CSV = PRIMARY_RESULTS_CSV
LOOKUP_CSV = DATA_DIR / "lesion_roi_lookup_1024x384.csv"
OUTDIR = OUT_DIR / "panknin_lesion_roi_check"

ALPHA = {1: 0.95, 2: 0.90, 3: 0.85, 4: 0.80, 5: 0.75, 6: 0.70}
DEGS = ["noise", "blur", "contrast", "jpeg2000", "resolution"]
TOL = 1e-3


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    R = pd.read_csv(RESULTS_CSV)
    R["image_id"] = R["image_id"].astype(str)
    LK = pd.read_csv(LOOKUP_CSV)
    LK["image_id"] = LK["image_id"].astype(str)
    clean = R[R.degradation_type == "clean"].set_index("image_id")

    c = R[(R.degradation_type == "contrast") & R.cnr.notna() & (R.label_45 == 1)].copy()
    c["alpha"] = c.severity.map(ALPHA)
    c["err"] = (c.delta_mu / c.image_id.map(clean["delta_mu"]) - c.alpha).abs()

    rows = []
    for iid, g in c.groupby("image_id"):
        slope, intercept = np.polyfit(g.alpha, g.delta_mu, 1)
        resid = float(np.max(np.abs(g.delta_mu - (slope * g.alpha + intercept))))
        rows.append(dict(image_id=iid, max_err=g.err.max(), flagged=g.err.max() > TOL,
                         slope=slope, intercept=intercept, affine_resid=resid,
                         clean_delta_mu=float(clean.loc[iid, "delta_mu"])))
    F = pd.DataFrame(rows).merge(LK, on="image_id", how="left")
    F["box_w"] = F.lesion_xmax - F.lesion_xmin
    F["box_h"] = F.lesion_ymax - F.lesion_ymin
    F["rel_intercept"] = F.intercept.abs() / F.clean_delta_mu.clip(lower=1e-12)
    F.to_csv(OUTDIR / "roi_mask_check.csv", index=False)

    bad = set(F.loc[F.flagged, "image_id"])
    ok, dev = F[~F.flagged], F[F.flagged]

    d = R[R.degradation_type != "clean"].copy()
    d["abs_dlogit"] = (d.logit - d.image_id.map(clean["logit"])).abs()
    pos = d[(d.label_45 == 1) & d.cnr.notna()]
    keep = pos[~pos.image_id.isin(bad)]

    L = ["Lesion-box / breast-mask asymmetry check", "",
         "flagged (delta_mu ratio off alpha by > %g): %d of %d images" % (TOL, len(dev), len(F)), "",
         "affine fit of delta_mu against alpha (6 contrast severities per image):",
         "  conforming: max |slope - clean delta_mu| = %.2e, median rel. intercept %.2e"
         % ((ok.slope - ok.clean_delta_mu).abs().max(), ok.rel_intercept.median()),
         "  flagged   : median rel. intercept %.3f, min %.2e" % (dev.rel_intercept.median(), dev.rel_intercept.min()),
         "",
         "lesion box size (px, canvas 1024x384):",
         "  conforming median %.0f x %.0f" % (ok.box_w.median(), ok.box_h.median()),
         "  flagged    median %.0f x %.0f" % (dev.box_w.median(), dev.box_h.median()),
         "",
         "impact: pooled Spearman with |dlogit|, all 192 -> 169 conforming"]
    for metric in ("cnr", "delta_mu"):
        L.append("  %s:" % metric)
        for g in DEGS:
            a, k = pos[pos.degradation_type == g], keep[keep.degradation_type == g]
            L.append("    %-12s %+0.3f -> %+0.3f" % (g, spearmanr(a[metric], a.abs_dlogit).correlation,
                                                     spearmanr(k[metric], k.abs_dlogit).correlation))
    rep = "\n".join(L)
    (OUTDIR / "roi_mask_report.txt").write_text(rep + "\n", encoding="utf-8")
    print(rep)


if __name__ == "__main__":
    main()
