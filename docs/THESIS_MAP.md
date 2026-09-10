# From the thesis to the code

Every figure, table and headline number in the thesis, and the script that produces it.

Two conventions run through this repository:

- `MAMMO_DATA_DIR` points at your copy of the released tables (default `./data`).
- `MAMMO_OUT_DIR` receives everything generated (default `./results`).

The raw mammograms come from [VinDr-Mammo](https://physionet.org/content/vindr-mammo/),
which requires credentialed access through PhysioNet. The released tables carry the model
outputs, the image quality metrics and the label columns, so the results can be recomputed
without the images.

---

## Start here: reproduce the results without a GPU

The per-image prediction tables in the data record hold one row per image x degradation x
severity, carrying the model logit and the image quality metrics for that same variant.
Every quantitative claim in the results chapter follows from them:

```bash
export MAMMO_DATA_DIR=/path/to/data
python reproduce/reproduce_from_predictions.py \
    --results-csv     $MAMMO_DATA_DIR/predictions/convnext_tiny_s42_degradation_results.csv \
    --multiarch-dir   $MAMMO_DATA_DIR/predictions \
    --val-predictions $MAMMO_DATA_DIR/predictions/val_clean_predictions.csv
```

It recomputes fifteen headline quantities and checks each against the value printed in the
thesis, exiting non-zero on any drift:

| quantity | thesis | where |
|---|---|---|
| Clean ROC-AUC | 0.824 | §5.1 |
| Clean average precision | 0.436 | §5.1 |
| JPEG 2000 S6 AUC / AP | 0.654 / 0.123 | §5.3 |
| JPEG 2000 S6 change in AUC | −0.170 | §5.3 |
| ECE, uncalibrated / Platt | 0.367 / 0.009 | Table 5.7 |
| Mean tau, clean / noise S6 | 0.0036 / 0.0281 | §5.2 |
| SSIM against prediction shift, JPEG 2000 | −0.821 | §5.4 |
| the same, within a fixed severity | −0.331 | §5.4 |
| Condition-level bridge, SSIM against AUC | +0.78 | §5.4 |
| Clean AUC by architecture | 0.828 / 0.782 / 0.753 | §5.3.3 |

Three definitions matter and are easy to get wrong.

1. **ECE uses 15 equally spaced bins** throughout. The uncalibrated value barely moves with the
   bin count, but the recalibrated one does, because Platt scaling concentrates the
   probabilities near the base rate. At 10 bins the recalibrated value reads 0.004, not 0.009.
2. **The correlations above are JPEG 2000-specific**, not pooled over all degradations. Pooling
   everything gives about -0.41, which the thesis does not claim. The +0.78 bridge is a
   different object again: one point per condition, condition-mean SSIM against the AUC achieved
   under that condition.
3. The released tables label the undegraded variant **`clean`**, not `none`, and motion blur
   **`blur`**, not `motion_blur`. The toolkit in `src/` uses the other names.

---

## Figures

Run these from inside `figures/`. They share `pubcommon.py`, which resolves paths through
`experiments/paths.py`.

| Figure | File | Script | Needs |
|---|---|---|---|
| 4.1 | `FIG_roi_definition` | `fig_roi_definition.py` | DICOM + QA tables |
| 4.2 | `FIG_degradation_diff_signed` | `make_degradation_diff.py` | DICOM + QA tables |
| 5.2 | `FIG_global_iqm_vs_severity` | `fig2_global_iqm.py` | predictions |
| 5.3 | `FIG_auc_auprc_vs_severity` | `fig3_4_studylevel.py` | predictions |
| 5.4 | `FIG_rq1_multiarch_auc_vs_severity` | `make_rq1_multiarch.py` | `multiarch_eval_tables/` |
| 5.5 | `FIG_global_iqm_prediction_shift_heatmap` | `fig5_6_bridge_heatmap.py` (`fig6`) | predictions |
| 5.6 | `FIG_iqm_performance_bridge_auc` | `fig5_6_bridge_heatmap.py` (`fig5`) | predictions |
| 5.7 | `FIG_reliability_classwise` | `make_a4_calibration.py` | predictions |
| 5.8 | `FIG_calibration_threshold_vs_severity` | `fig7_8_dist_calibration.py` | predictions |
| A | `FIG_degradation_examples_full` / `_zoom` | `fig1_degradation_examples.py`, `fig_degradation_examples_tight.py` | DICOM |
| A | `FIG_degradation_diff_abs` | `make_degradation_diff.py` | DICOM |
| A | `FIG_delta_auc_auprc_forest` | `fig3_4_studylevel.py` | predictions |
| A | `FIG_score_distributions_clean_vs_severe` | `fig7_8_dist_calibration.py` | predictions |
| A | `FIG_calibration_temperature_scaled` | `fig8b_temperature_scaling.py` | predictions + validation |
| A | `figure2_display_params_by_manufacturer` | `fig_display_params_manufacturer_pdf.py` | DICOM metadata table |
| A | `APP_*` panels | `fig5_6_bridge_heatmap.py`, `fig7_8_dist_calibration.py`, `fig3_4_robustness.py` | predictions |

Run `precompute_metrics.py` once first: it writes `condition_metrics_full.csv`, which several
figure scripts read.

Two notes. `fig3_4_studylevel.py` bootstraps study-level confidence intervals and takes about
ten minutes; `_fig53_iter.py` re-renders it from a cached pickle in seconds while tuning
layout. Figure 2.1 is a raster from the IAEA and has no generating script.

---

## Experiment code

| Stage | Script |
|---|---|
| Degradation operators, 5 types x 6 severities | `experiments/degradations.py` |
| Evaluation: degrade, infer, compute metrics per variant | `experiments/evaluate_degradations.py` |
| Aggregate into per-condition metrics | `experiments/analyze_degradation_results.py` |
| Lesion boxes mapped onto the working canvas | `experiments/build_lesion_roi_lookup.py` |
| Masked SSIM and reference tau, as reported | `experiments/recompute_ssim_tau.py` |
| Training, all three architectures | `experiments/training/mammo-18-v3.py` |
| Job scripts as submitted | `experiments/slurm/*.slurm` |

`experiments/panknin/` holds four scripts written to answer specific examiner questions: an
alternative performance measure, lesion-level pooled correlations, a check that the lesion ROI
stays valid under motion blur, and the physical anchoring of the severity ladders.

### Degradation parameters

Fixed in `degradations.py` and reproduced in `configs/default.yaml`:

| Degradation | S1 | S2 | S3 | S4 | S5 | S6 |
|---|---|---|---|---|---|---|
| Dose-motivated noise, `f_D` | 0.95 | 0.85 | 0.75 | 0.65 | 0.50 | 0.35 |
| Motion blur, kernel px | 3 | 5 | 7 | 11 | 15 | 21 |
| Contrast, `alpha` | 0.95 | 0.90 | 0.85 | 0.80 | 0.75 | 0.70 |
| JPEG 2000, CR | 10 | 25 | 50 | 100 | 250 | 500 |
| Resolution, `r` | 0.90 | 0.80 | 0.70 | 0.60 | 0.50 | 0.40 |

Motion blur is specified in pixels, not millimetres. A millimetre-first schedule collapses at
the sub-0.1 mm pixel spacings of this dataset: several severities round to the same odd kernel
and produce identical images. On the 1024x384 working canvas one pixel spans a median of
0.240 mm, so the kernels correspond to median motion extents of 0.72, 1.20, 1.68, 2.64, 3.60
and 5.04 mm in the patient plane.

---

## Model weights

Nine checkpoints, three architectures x three seeds, in the data record. Each stores only
`model_state_dict`; the validation AUC is in the filename, and the checkpoint kept per run is
the one with the highest validation AUC.

| Architecture | Parameters | Seeds 42 / 43 / 44, validation AUC | Mean clean test AUC |
|---|---|---|---|
| ConvNeXt-Tiny | 27.82 M | 0.8103 / 0.8161 / 0.8292 | 0.828 |
| EfficientNet-B4 | 17.68 M | 0.7677 / 0.8147 / 0.7717 | 0.782 |
| ResNet-18 | 11.19 M | 0.7952 / 0.8023 / 0.7751 | 0.753 |

ConvNeXt-Tiny seed 42 is the primary model throughout the thesis: clean test AUC 0.824,
average precision 0.436, ECE 0.367.

To regenerate predictions from a checkpoint rather than using the released tables, use
`experiments/evaluate_degradations.py`, which is the same entry point the released tables came
from. `experiments/slurm/evaluate_degradations.slurm` runs it across all nine.

---

## A caveat on the metric columns

The primary model's table carries the metric definitions used throughout the thesis: SSIM
restricted to the breast mask, the reference implementation of tau, and `contrast_iqr`. The
other eight tables come straight from `evaluate_degradations.py` and carry **unmasked SSIM and
the in-script tau**. Predictions are unaffected, so every AUC, average-precision, calibration
and threshold result is directly comparable across all nine models; only the image quality
columns differ in definition. `experiments/recompute_ssim_tau.py` produces the thesis
definitions from an evaluation output.

## Data

The prediction tables ship complete, including `label_45`, `label_5`, `density` and
`manufacturer`, so no join is needed. `reproduce/join_labels.py` remains available for tables
that lack them, and it also documents how `label_45` is derived: BI-RADS 4 or 5 against 1 to 3.

The split manifest is in the data record and can also be regenerated deterministically with
`experiments/build_complete_splits_csv.py` (`StratifiedShuffleSplit`, seed 42, stratified on
label and manufacturer). Checksums for the two frozen tables:

| File | SHA-256 |
|---|---|
| `master_splits_1024x384_complete.csv` | `e8ef4f7c56797e1913281aff39b392b378df63f003a6235922e05b2c255a1b8f` |
| `lesion_roi_lookup_1024x384.csv` | `00e91c288dd49e951e2dc53d90aac7cfcd93e3e948e9678102de225bdfc0bb46` |

The split is 12,800 training, 3,200 validation and 4,000 test images over 20,000 images and
5,000 studies, with 630, 160 and 198 positives respectively. The mammograms themselves come from
PhysioNet under credentialed access and are not redistributed.

---

## Software environment

Training and evaluation ran on the PTB high-performance cluster under SLURM, in the conda
environment `mammo_torch`: **Python 3.11.15, torch 2.5.1+cu121, torchvision 0.20.1+cu121**,
numpy 2.4.6, scipy 1.17.1, scikit-learn 1.8.0, scikit-image 0.26.0, pandas 3.0.2,
opencv-python-headless 4.13.0.92, matplotlib 3.10.9, glymur (OpenJPEG backend), pydicom,
wandb 0.27.0, webdataset 1.0.2. Pretrained ImageNet-1K weights came from `torchvision`
(`ResNet18_Weights.DEFAULT`, `EfficientNet_B4_Weights.DEFAULT`,
`ConvNeXt_Tiny_Weights.DEFAULT`).
