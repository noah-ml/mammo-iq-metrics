#!/usr/bin/env python3
"""Sensitivity analysis: prediction shift vs. a per-image accuracy change.

Background
----------
The second examiner (Panknin, annotation batch 3, 2026-08-20) asked repeatedly
why RQ2 correlates the image quality metrics with the per-image *prediction
shift* |dl| = |logit_deg - logit_clean| rather than with a quantity that says
whether the prediction got *worse*. He proposed

    d_acc_i = |y_i - p_clean,i| - |y_i - p_deg,i|      (>0: degraded is closer to y)

This script recomputes the whole RQ2 correlation matrix with his measure in
place of |dl| and reports how much the answer changes. It uses the same
per-image results CSV as every published figure (ConvNeXt-Tiny no-ColorJitter,
training job 956694 / degradation eval 957382), so no model inference is
repeated.

Convention used here: err_shift = |y - p_deg| - |y - p_clean|, i.e. the sign is
flipped relative to his formula so that POSITIVE means the prediction moved
AWAY from the label (worse). The unsigned |err_shift| is the like-for-like
counterpart of |dl|, which is itself unsigned.

Outputs
-------
  panknin_alt_measure_20260820/alt_measure_correlations.csv
  panknin_alt_measure_20260820/alt_measure_report.txt
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
OUTDIR = OUT_DIR / "panknin_alt_measure"

IQMS = ["ssim", "tenengrad", "noise_var", "tau", "contrast_iqr"]
IQM_LABEL = {"ssim": "SSIM", "tenengrad": "Tenengrad", "noise_var": "Noise var.",
             "tau": "tau", "contrast_iqr": "IQR"}
DEGS = ["noise", "blur", "contrast", "jpeg2000", "resolution"]
DEG_LABEL = {"noise": "Dose-motivated noise", "blur": "Motion blur",
             "contrast": "Contrast reduction", "jpeg2000": "JPEG2000",
             "resolution": "Resolution reduction"}


def load() -> pd.DataFrame:
    R = pd.read_csv(RESULTS_CSV)
    R["image_id"] = R["image_id"].astype(str)
    clean = R[R.degradation_type == "clean"].set_index("image_id")
    deg = R[R.degradation_type != "clean"].copy()
    deg["clean_logit"] = deg.image_id.map(clean["logit"])
    deg["clean_prob"] = deg.image_id.map(clean["prob"])
    y = deg.label_45.astype(float)
    deg["abs_dlogit"] = (deg.logit - deg.clean_logit).abs()
    deg["err_shift"] = (y - deg.prob).abs() - (y - deg.clean_prob).abs()
    deg["abs_err_shift"] = deg.err_shift.abs()
    return deg


def matrix(deg: pd.DataFrame, target: str) -> pd.DataFrame:
    out = {}
    for m in IQMS:
        col = {}
        for d in DEGS:
            sub = deg[deg.degradation_type == d][[m, target]].dropna()
            col[d] = spearmanr(sub[m], sub[target]).correlation
        out[m] = col
    return pd.DataFrame(out).loc[DEGS, IQMS]


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    deg = load()
    n_per = deg.groupby("degradation_type").size().unique()

    m_shift = matrix(deg, "abs_dlogit")
    m_acc = matrix(deg, "abs_err_shift")
    m_signed = matrix(deg, "err_shift")
    diff = (m_acc - m_shift).abs()

    long = []
    for name, M in (("abs_dlogit", m_shift), ("abs_err_shift", m_acc),
                    ("signed_err_shift", m_signed)):
        for d in DEGS:
            for m in IQMS:
                long.append(dict(measure=name, degradation=d, iqm=m,
                                 rho=round(float(M.loc[d, m]), 4)))
    pd.DataFrame(long).to_csv(OUTDIR / "alt_measure_correlations.csv", index=False)

    mean_shift = deg.pivot_table(index="degradation_type", columns="severity",
                                 values="err_shift", aggfunc="mean").loc[DEGS]

    lines = []
    add = lines.append
    add("Sensitivity analysis: |delta logit| vs. per-image accuracy change")
    add("Source: %s" % RESULTS_CSV.name)
    add("n per degradation: %s image-severity observations" % ", ".join(map(str, n_per)))
    add("")
    add("Spearman rho, IQM vs |delta logit| (as published, Table A.9 / Figure 5.5)")
    add(m_shift.round(3).to_string())
    add("")
    add("Spearman rho, IQM vs |change in absolute error| (examiner's measure)")
    add(m_acc.round(3).to_string())
    add("")
    add("Spearman rho, IQM vs SIGNED change in absolute error")
    add(m_signed.round(3).to_string())
    add("")
    add("Absolute difference between the first two matrices:")
    add("  max  = %.3f  (%s)" % (diff.max().max(),
                                 diff.stack().idxmax()))
    add("  mean = %.3f" % diff.values.mean())
    add("  rank agreement of the 25 cells (Spearman) = %.3f" %
        spearmanr(m_shift.values.ravel(), m_acc.values.ravel()).correlation)
    add("  strongest correlate per degradation identical: %s" %
        bool((m_shift.abs().idxmax(axis=1) == m_acc.abs().idxmax(axis=1)).all()))
    add("")
    add("Mean signed error shift (>0 = prediction moved away from the label):")
    add(mean_shift.round(3).to_string())
    report = "\n".join(lines)
    (OUTDIR / "alt_measure_report.txt").write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
