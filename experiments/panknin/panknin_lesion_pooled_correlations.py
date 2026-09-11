#!/usr/bin/env python3
"""Lesion-level metrics: severity-6 snapshot and pooled severity 1-6 correlations.

Background
----------
The second examiner (Panknin, annotation batch 3, 2026-08-20) asked why the
lesion-level analysis excluded four of the five degradations at the milder
severities. Checking the data showed the premise of the thesis sentence was
wrong: cnr and delta_mu are present for all 357 annotated images in all 30
degraded conditions, in every results CSV, because `evaluate_degradations.py`
computes them per variant from ROIs fixed once on the clean image
(`build_lesion_roi_lookup.py`), which makes the lookup severity-independent by
construction.

This script regenerates the numbers for Table 5.5 from the same per-image CSV
that feeds every published figure (`pubcommon.RESULTS_CSV`), for both the
severity-6 snapshot and the pooled severities 1-6, so the table is reproducible
from the released data.

Output
------
  panknin_lesion_pooled_20260820/lesion_correlations.csv
  panknin_lesion_pooled_20260820/lesion_report.txt
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import DATA_DIR, OUT_DIR, PRIMARY_RESULTS_CSV, require  # noqa: E402

RESULTS_CSV = PRIMARY_RESULTS_CSV
OUTDIR = OUT_DIR / "panknin_lesion_pooled"

DEGS = ["noise", "blur", "contrast", "jpeg2000", "resolution"]
LABEL = {"noise": "Dose-motivated noise", "blur": "Motion blur",
         "contrast": "Contrast reduction", "jpeg2000": "JPEG2000",
         "resolution": "Resolution reduction"}


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    R = pd.read_csv(RESULTS_CSV)
    R["image_id"] = R["image_id"].astype(str)
    clean = R[R.degradation_type == "clean"].set_index("image_id")
    d = R[R.degradation_type != "clean"].copy()
    d["abs_dlogit"] = (d.logit - d.image_id.map(clean["logit"])).abs()

    # positive (BI-RADS 4+5) annotated cohort used for the RQ2 lesion analysis
    pos = d[(d.label_45 == 1) & d.cnr.notna()].copy()
    # sigma_Bkg is not stored directly; it is delta_mu / cnr by construction
    pos["sigma_bkg"] = pos.delta_mu / pos.cnr
    cl_sigma = (clean.delta_mu / clean.cnr)
    pos["dmu_ratio"] = pos.delta_mu / pos.image_id.map(clean["delta_mu"])
    pos["cnr_ratio"] = pos.cnr / pos.image_id.map(clean["cnr"])
    pos["sigma_ratio"] = pos.sigma_bkg / pos.image_id.map(cl_sigma)

    rows = []
    for g in DEGS:
        sub = pos[pos.degradation_type == g]
        s6 = sub[sub.severity == 6]
        rows.append(dict(
            degradation=LABEL[g],
            dmu_ratio_s6=s6.dmu_ratio.median(),
            sigma_ratio_s6=s6.sigma_ratio.median(),
            cnr_ratio_s6=s6.cnr_ratio.median(),
            rho_cnr_s6=spearmanr(s6.cnr, s6.abs_dlogit).correlation,
            rho_dmu_s6=spearmanr(s6.delta_mu, s6.abs_dlogit).correlation,
            rho_cnr_pooled=spearmanr(sub.cnr, sub.abs_dlogit).correlation,
            rho_dmu_pooled=spearmanr(sub.delta_mu, sub.abs_dlogit).correlation,
            n_s6=len(s6), n_pooled=len(sub)))
    T = pd.DataFrame(rows)
    T.to_csv(OUTDIR / "lesion_correlations.csv", index=False)

    lines = ["Lesion-level metrics, positive annotated cohort (n = %d images)" % pos.image_id.nunique(),
             "Source: %s" % RESULTS_CSV.name, "",
             T.round(3).to_string(index=False), "",
             "LaTeX rows for Table 5.5 (ratios | rho s6 | rho pooled):"]
    for r in rows:
        lines.append("        %-22s & %.3f & %.3f & %.3f & $%+0.3f$ & $%+0.3f$ & $%+0.3f$ & $%+0.3f$ \\\\"
                     % (r["degradation"], r["dmu_ratio_s6"], r["sigma_ratio_s6"], r["cnr_ratio_s6"],
                        r["rho_cnr_s6"], r["rho_dmu_s6"], r["rho_cnr_pooled"], r["rho_dmu_pooled"]))
    report = "\n".join(lines)
    (OUTDIR / "lesion_report.txt").write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
