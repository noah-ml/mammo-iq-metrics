#!/usr/bin/env python3
"""Write thesis-ready caption .txt files and the README.md."""
from __future__ import annotations
import pubcommon as pc

NB = pc.N_BOOT
COMMON = ("Model: ConvNeXt-Tiny (no ColorJitter), training job 956694, degradation eval with the "
          "added τ IQM (W&B run convnext-tiny-…-nojitter-tau-strict-clean; identical to run 957382 "
          "plus the τ column, clean AUC=0.824/AUPRC=0.436 unchanged). Test set n=4000 images "
          "(198 positive for label_45 = BI-RADS 4–5). Degradation severities: 0 = clean, 1–6 "
          "increasing. No model inference was repeated; all values derive from the stored "
          "per-image per-condition predictions and IQMs. τ is the nonparametric noise measure of "
          "Anton et al. (Phys. Med. Biol. 68:045003, 2023), computed on the breast mask.")

CAPS = {
"FIG_degradation_examples": f"""
Figure 1. Simulated image-quality degradations at increasing severity. One representative
positive mammogram (cropped to the breast in the full version; zoomed to the lesion region in
the companion version) is shown clean and at severities 1, 3 and 6 for each of the five
degradation types (dose noise, horizontal motion blur, contrast reduction, JPEG2000 compression,
resolution reduction). Intensity windowing is identical across all panels ([0,1]). The SSIM of
each degraded image relative to its clean version is annotated beneath each panel. The figure
illustrates what each severity level means perceptually: dose noise and JPEG2000 produce the
largest pixel-level changes (lowest SSIM), motion blur and resolution loss visibly smear
structure, while contrast reduction is perceptually mild (SSIM stays high) — consistent with its
near-flat effect on model performance. Illustrative single case; not a quantitative result.
""",
"FIG_global_iqm_vs_severity": f"""
Figure 2. Global image-quality metrics versus severity, all test images. Mean (A) SSIM
(full-reference, vs clean), (B) Tenengrad (gradient-energy sharpness), (C) noise variance
(homogeneous-patch) and (D) τ (Anton et al. nonparametric noise measure, breast-mask), one line
per degradation; shaded bands are 95% bootstrap confidence intervals of the mean ({NB} resamples
over images). {COMMON} Key result: SSIM decreases monotonically for every degradation (validating
that the simulations measurably reduce image quality), most steeply for JPEG2000 and noise.
Tenengrad falls for blur/resolution/JPEG2000 but rises for dose noise (noise injects spurious
high-frequency gradients). Noise variance and τ both increase specifically for dose noise (τ rises
≈3.6× from clean to severity 6) and are largely unresponsive to the other artifacts, confirming
they are noise-specific IQMs; τ additionally decreases under blur and contrast (both reduce
local pixel-to-pixel differences), consistent with its construction. Bands are very narrow because
n=4000.
""",
"FIG_auc_auprc_vs_severity": f"""
Figure 3. Classifier robustness versus severity. (A) ROC-AUC and (B) average precision (AUPRC)
as a function of severity, one line per degradation; shaded bands are 95% bootstrap CIs ({NB}
image-resamples). Dashed line = clean baseline (AUC=0.824, AUPRC=0.437); dotted line = baseline
− 0.05. {COMMON} AUPRC is reported alongside AUC because the test set is highly imbalanced
(198/4000 positive), where AUPRC is the more sensitive operating measure. Key result: JPEG2000 is
the most damaging (AUC −0.17, AUPRC −0.31 at severity 6), followed by noise, resolution and blur;
contrast reduction stays on the clean baseline throughout. AUPRC degrades proportionally far more
than AUC for every degradation, underlining that ranking quality for the rare positive class is
hit harder than overall separability.
""",
"FIG_delta_auc_auprc_forest": f"""
Figure 4. Performance drop at maximum severity (severity 6). Forest plot of (A) ΔAUC and
(B) ΔAUPRC relative to clean, one row per degradation, sorted worst-to-best; markers colored by
degradation, horizontal bars are 95% bootstrap CIs ({NB} resamples). Solid line = no change;
dotted line = −0.05 practical-drop reference. {COMMON} Key result: JPEG2000 shows the largest and
most clearly non-trivial drop (ΔAUC ≈ −0.17), while contrast reduction’s CI overlaps zero
(ΔAUC ≈ −0.005), i.e. no detectable effect at the tested severities. Effect sizes, not p-values,
are the focus here.
""",
"FIG_iqm_performance_bridge_auc": f"""
Figure 5. Image-quality-to-performance bridge. Each marker is one degradation×severity condition;
lines trace severity 0→6 within each degradation; the star marks the shared clean condition. The
y-axis is condition-level ROC-AUC; x-axes are mean (A) SSIM, (B) Tenengrad, (C) noise variance,
(D) τ (Anton et al.). Subtitles give the pooled Spearman ρ across the 30 degraded conditions.
{COMMON} Key result: mean SSIM is the strongest aggregate bridge to AUC (monotone, high |ρ|):
conditions with lower mean SSIM have lower AUC across degradation types. Tenengrad is informative
but degradation-specific (noise breaks the monotone trend); mean noise variance and τ are
noise-specific and do not generalize as pooled predictors across degradation types. Correlations
are over aggregate conditions (n=30), so they are descriptive summaries, not tests over independent
images.
""",
"FIG_global_iqm_prediction_shift_heatmap": f"""
Figure 6. Global IQMs versus image-level prediction instability. Spearman ρ between each global
IQM (SSIM, Tenengrad, noise variance, τ) and the absolute logit shift |Δlogit| = |logit_degraded −
logit_clean|, computed per degradation over all test images and severities 1–6 (diverging colormap
centered at 0; *p<0.05, **p<0.01, ***p<0.001). {COMMON} Key result: SSIM is the dominant
correlate of prediction instability for every degradation (strongest for JPEG2000, ρ=−0.83):
lower SSIM ⇒ larger prediction shift. Tenengrad is secondary; noise variance and τ track
prediction instability specifically for dose noise (τ being a dedicated noise measure), with
little cross-degradation generalization. Caveat: severities are pooled and multiple severities
share the same source image, so the observations are not independent — p-values are
descriptive/exploratory and the large pooled n makes even tiny correlations 'significant';
interpret by effect size.
""",
"FIG_score_distributions_clean_vs_severe": f"""
Figure 7. Logit distributions explaining the AUC drop. Per-class logit histograms (blue =
negative, red = positive; density-normalized, shared x-axis) for clean and the severity-6
conditions of JPEG2000, motion blur, contrast and resolution; the dashed line is the clean Youden
decision threshold. Each panel is annotated with its AUC and AUPRC. {COMMON} Key result: under
severe JPEG2000, blur and resolution the two class distributions shift and overlap far more than
on clean (and the whole distribution drifts above threshold, driving sensitivity→1/specificity→0),
which is the mechanism behind the AUC/AUPRC loss; contrast severity 6 leaves the distributions
almost identical to clean, consistent with its flat performance curve.
""",
"FIG_calibration_threshold_vs_severity": f"""
Figure 8. Calibration and fixed-threshold behavior versus severity. (A) Expected calibration error
(ECE, 15 bins), (B) Brier score, (C) sensitivity and (D) specificity, the latter two at the fixed
clean Youden threshold (0.449, determined once on the clean validation set and held constant). One line
per degradation; dashed line = clean value. {COMMON} Key result: degradation worsens calibration
(higher ECE/Brier) for all but contrast; at the fixed threshold, severe blur/JPEG2000/resolution
push predictions upward so sensitivity rises toward 1 while specificity collapses toward 0 — the
model labels almost everything positive. Note: the absolute calibration is poor even on clean
(ECE≈0.37) because training used heavy positive class weighting (pos_weight≈19), which inflates
predicted probabilities; the degradation-induced *change* is the robustness signal of interest.
""",
}

APP_NOTE = """
Appendix figures.
APP_delta_auc_all_severities — ΔAUC for severities 1–6, all degradations (companion to Fig 4).
APP_iqm_performance_bridge_auprc — Fig 5 bridge against AUPRC instead of AUC.
APP_global_iqm_prediction_shift_scattergrid — per-(degradation×IQM) scatter of |Δlogit| colored by
  severity with a binned-median trend (underlying detail for the Fig 6 heatmap).
APP_roc_curves_clean_s3_s6 / APP_pr_curves_clean_s3_s6 — full ROC and PR curves at clean / severity
  3 / severity 6 for each degradation.
APP_f1_balacc_vs_severity — F1 and balanced accuracy at the fixed clean threshold vs severity.
APP_reliability_clean_jpeg_contrast — reliability diagrams (clean vs JPEG2000 s6 vs contrast s6).
All use the same model/run/threshold and the same descriptive-statistics caveats as the main text.
"""

README = f"""# Thesis publication figures — mammography robustness

{COMMON}

All figures are saved as **.pdf** (vector, for LaTeX `\\includegraphics`) and **.png** (300 dpi
preview). Captions are in `captions/` (use as `\\caption{{}}`; they are intentionally NOT embedded
in the figures). Underlying numbers are in `tables/`. Every figure is reproduced by the scripts in
`scripts/` from the single results CSV (no-jitter τ run, training job 956694; = run 957382 plus the
τ IQM column) — no model inference is repeated. The lesion
CNR/Δμ composite figure (FIG_cnr_deltamu_rq2_composite) already exists in the separate QA folder
and is referenced, not duplicated.

## Suggested main-text figure order
1. **FIG_degradation_examples** (full + zoom) — what the degradations look like.
2. **FIG_global_iqm_vs_severity** — the simulations measurably change global IQMs (validation).
3. **FIG_auc_auprc_vs_severity** — headline robustness result (AUC + AUPRC).
4. **FIG_delta_auc_auprc_forest** — drop at severity 6 at a glance.
5. **FIG_iqm_performance_bridge_auc** — do average IQMs explain condition-level performance? (SSIM yes)
6. **FIG_global_iqm_prediction_shift_heatmap** — IQMs vs image-level prediction instability.
7. **FIG_cnr_deltamu_rq2_composite** — (already created) lesion-based CNR/Δμ; CNR invariant under
   linear contrast, Δμ scales with α; RQ2 positives-only.
8. **FIG_score_distributions_clean_vs_severe** and/or **FIG_calibration_threshold_vs_severity** —
   mechanism (score overlap) and calibration/threshold robustness; include as space allows.

## One-paragraph interpretation per figure
- **Fig 1:** Severity levels are perceptually meaningful; noise/JPEG2000 change pixels most,
  contrast least.
- **Fig 2:** SSIM falls for all degradations (validates the pipeline); Tenengrad rises under noise;
  noise variance is noise-specific.
- **Fig 3:** JPEG2000 worst, contrast flat; AUPRC drops far more than AUC on this imbalanced set.
- **Fig 4:** Severity-6 ΔAUC/ΔAUPRC ranking; contrast’s CI includes 0 (no effect), JPEG2000 worst.
- **Fig 5:** Mean SSIM is the best aggregate bridge to AUC; Tenengrad degradation-specific; noise
  variance does not generalize.
- **Fig 6:** SSIM most strongly tracks |Δlogit| for every degradation (descriptive ρ; pooled n).
- **Fig 7:** Severe blur/JPEG2000/resolution increase positive–negative score overlap and shift
  scores above threshold; contrast barely moves them.
- **Fig 8:** Degradation worsens calibration and destroys fixed-threshold specificity (except
  contrast); absolute calibration is poor even on clean due to class-weighted training.

## Appendix figures
{APP_NOTE}

## Interpretation rules applied throughout
- Global performance + global IQMs use all 4000 test images; lesion CNR/Δμ use positive annotated
  cases only (see the separate QA folder).
- Severity-pooled correlations treat multiple severities of the same image, so they are
  **descriptive/exploratory**, not tests over independent samples; p-values are reported as such.
- Contrast reduction: CNR is **analytically invariant** under the implemented linear transform
  (numerator Δμ and denominator σ_bg both scale by α); Δμ scales with α. We do **not** claim CNR
  should fall.
- We emphasize **effect sizes and practical drops** (e.g. the −0.05 AUC reference) over significance
  stars, especially given the large pooled n.

## Tables (`tables/`)
- `condition_metrics_full.csv/.md` — AUC, AP, sensitivity, specificity, F1, balanced accuracy, ECE,
  Brier per degradation×severity (fixed clean Youden threshold = 0.449, fixed on validation).
- `robustness_auc_auprc_table.csv`, `delta_performance_table.csv` — Fig 3/4 values + bootstrap CIs.
- `global_iqm_vs_severity.csv` — Fig 2 means/medians + CIs.
- `bridge_condition_table.csv` — Fig 5 condition-level table.
- `global_iqm_prediction_shift_correlations.csv` — Fig 6 ρ/p.
- `calibration_threshold_table.csv` — Fig 8 values.
"""


def main():
    for name, text in CAPS.items():
        pc.write_caption(name, text)
    pc.write_caption("APPENDIX_figures", APP_NOTE)
    (pc.PUBDIR / "README.md").write_text(README, encoding="utf-8")
    print(f"wrote {len(CAPS)+1} caption files and README.md")


if __name__ == "__main__":
    main()
