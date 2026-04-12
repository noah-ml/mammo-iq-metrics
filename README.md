# Mammography Image Quality & Neural Network Robustness

Quantitative analysis of how image quality degradation affects deep learning model performance in mammography screening.

> **MSc Thesis** · Biomedical Engineering, TU Berlin (2026)  
> In cooperation with the Physikalisch-Technische Bundesanstalt (PTB)

## Overview

This pipeline systematically quantifies how defined image quality dimensions affect the performance (AUC), calibration, and sensitivity of deep learning models for mammography. The approach:

1. Load full-resolution DICOM mammograms from VinDr-Mammo
2. Compute a breast mask and crop to the breast region
3. Fix ROIs (homogeneous tissue patch, lesion ring) on the **original** image
4. Apply parametrized degradations to the normalized image
5. Compute image quality metrics on degraded variants using the **same fixed ROIs**
6. Feed degraded images + IQM values into model training/inference to produce performance-degradation curves

## Degradations

All three degradation types use physically calibrated parameters with 6 severity levels.

### Dose-based noise
Simulates reduced X-ray dose. `dose_factor` is the fraction of original dose; noise is added to match the expected increase in quantum noise.

| Severity | `dose_factor` |
|---|---|
| 1 | 0.95 |
| 2 | 0.85 |
| 3 | 0.75 |
| 4 | 0.65 |
| 5 | 0.50 |
| 6 | 0.35 |

### Motion blur
Horizontal rectangular PSF simulating detector/patient motion. Length is specified in mm and converted to pixels via `PixelSpacing` from the DICOM header (fallback for ~0.07 mm/px).

| Severity | `length_mm` | `length_px` (fallback) |
|---|---|---|
| 1 | 0.10 | 3 |
| 2 | 0.25 | 5 |
| 3 | 0.50 | 7 |
| 4 | 0.75 | 11 |
| 5 | 1.00 | 15 |
| 6 | 1.50 | 21 |

### Contrast compression
Linear contrast reduction centered on the masked tissue median: `x_out = α·(x_in − center) + center`.

| Severity | `alpha` |
|---|---|
| 1 | 0.95 |
| 2 | 0.90 |
| 3 | 0.85 |
| 4 | 0.80 |
| 5 | 0.75 |
| 6 | 0.70 |

Each batch run produces **19 variants per image**: 1 baseline + 6 noise + 6 motion blur + 6 contrast.

## Metrics

**All images:**
| Metric | Description |
|---|---|
| `noise_variance` | Variance of a fixed homogeneous tissue patch |
| `tenengrad` | Mean squared Sobel gradient over breast mask (sharpness) |
| `ssim` | Full-reference SSIM vs. original, masked to breast region |

**Lesion images only** (requires bounding box annotation):
| Metric | Description |
|---|---|
| `cnr` | \|lesion\_mean − bg\_mean\| / bg\_std (background ring) |
| `delta_mu` | \|lesion\_mean − bg\_mean\| (absolute contrast) |

ROIs are always determined on the **original image** and reused for all degraded variants to ensure fair comparison.

## Project Structure

```
├── src/
│   ├── roi_metrics.py      # Core DICOM-first toolkit: masking, ROI selection, metrics, degradations
│   ├── test_runner.py      # Single-image and batch-list experiment runner
│   └── run_batch.py        # Stratified 50-sample batch pipeline
├── configs/
│   └── default.yaml        # Pipeline configuration and degradation plan
├── tests/
│   └── test_metrics.py     # Unit tests for metrics and degradations
├── requirements.txt
└── README.md
```

## Quickstart

```bash
git clone https://github.com/noah-ml/mammo-iq-metrics.git
cd mammo-iq-metrics
pip install -r requirements.txt
```

**Single image:**
```bash
python src/test_runner.py \
  --image path/to/image.dicom \
  --annotations path/to/finding_annotations.csv \
  --output-dir results/single
```

**Single image with one specific degradation (severity 1–6):**
```bash
python src/test_runner.py \
  --image path/to/image.dicom \
  --degradation motion_blur --deg-severity 4 \
  --output-dir results/single
```

**50-sample stratified batch** (default: 40 with lesion + 10 without):
```bash
export VINDR_ROOT=/path/to/vindr-mammo-dataset
python src/run_batch.py --output-dir results/batch_50
```

**Custom sample counts or seed:**
```bash
python src/run_batch.py \
  --n-with-lesion 40 --n-without-lesion 10 \
  --seed 2024 --output-dir results/batch_50
```

**Run tests:**
```bash
pytest tests/ -v
```

## Dataset

[VinDr-Mammo](https://vindr.ai/datasets/mammo) — 5,000 full-field digital mammography studies (4 views each) with BI-RADS and finding-level bounding box annotations. Available via PhysioNet.

Expected layout:
```
<VINDR_ROOT>/
├── images/<study_id>/<image_id>.dicom
├── finding_annotations.csv
└── breast-level_annotations.csv
```

## Output

Each batch run produces:
- `metrics_batch.csv` — one row per image × degradation variant
- `run_summary.json` — config, per-image summaries, error log
- `<image_id>/` — QC overlay PNGs and per-image `_summary.json`

## Tech Stack

Python · pydicom · OpenCV · scikit-image · NumPy · SciPy · Pillow · matplotlib

## License

MIT
