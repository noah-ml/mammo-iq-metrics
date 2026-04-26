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

All five degradation types use physically calibrated parameters with 6 severity levels each.

### Dose-based noise
Simulates reduced X-ray dose. `dose_factor` is the fraction of original dose; noise is added to match the expected increase in quantum noise variance (∝ 1/dose).

| Severity | `dose_factor` |
|---|---|
| 1 | 0.95 |
| 2 | 0.85 |
| 3 | 0.75 |
| 4 | 0.65 |
| 5 | 0.50 |
| 6 | 0.35 |

### Motion blur
Horizontal rectangular PSF simulating patient/detector motion during exposure. Length is specified in mm and converted to pixels via `PixelSpacing` from the DICOM header. Values are chosen so each severity level maps to a **distinct odd kernel size** at the VinDr-Mammo pixel spacing of 0.085 mm/px.

| Severity | `length_mm` | `kernel_px` (at 0.085 mm/px) | `length_px` (fallback) |
|---|---|---|---|
| 1 | 0.26 | 3 | 3 |
| 2 | 0.43 | 5 | 5 |
| 3 | 0.60 | 7 | 7 |
| 4 | 0.77 | 9 | 9 |
| 5 | 0.94 | 11 | 11 |
| 6 | 1.45 | 17 | 17 |

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

### JPEG 2000 compression
Lossy wavelet compression at increasing compression ratios. CR=10 is clinically acceptable; CR=500 causes severe detail loss.

| Severity | `compression_ratio` |
|---|---|
| 1 | 10 |
| 2 | 25 |
| 3 | 50 |
| 4 | 100 |
| 5 | 250 |
| 6 | 500 |

### Spatial resolution loss
Downscale with `INTER_AREA` then upscale back to original size with `INTER_LINEAR`, simulating reduced detector resolution or pixel binning.

| Severity | `scale_factor` |
|---|---|
| 1 | 0.90 |
| 2 | 0.75 |
| 3 | 0.60 |
| 4 | 0.50 |
| 5 | 0.40 |
| 6 | 0.33 |

Each batch run produces **31 variants per image**: 1 baseline + 6 noise + 6 motion blur + 6 contrast + 6 JPEG 2000 + 6 resolution.

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

## Docker

**Build the image:**
```bash
docker compose build
```

**Run the full-dataset pipeline** (writes results to `./results/`):
```bash
# Set VINDR_ROOT to your local dataset path, then:
docker compose run --rm iqm-pipeline
# or override workers/output inline:
docker compose run --rm iqm-pipeline \
  src/compute_iqms_full_dataset.py --workers 4 --output-dir /app/results
```

**Run the 50-sample batch only:**
```bash
docker compose run --rm batch
```

**Single image:**
```bash
docker compose run --rm iqm-pipeline \
  src/test_runner.py \
  --image /data/vindr-mammo/images/<study_id>/<image_id>.dicom \
  --annotations /data/vindr-mammo/finding_annotations.csv \
  --output-dir /app/results/single
```

The dataset directory is mounted read-only at `/data/vindr-mammo` inside the container. Set the `VINDR_ROOT` environment variable on your host to point to your local copy, or edit the `docker-compose.yml` volume path directly.

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

[VinDr-Mammo](https://vindr.ai/datasets/mammo): 5,000 full-field digital mammography studies (4 views each) with BI-RADS and finding-level bounding box annotations. Available via PhysioNet.

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

Python · pydicom · OpenCV · scikit-image · NumPy · SciPy · Pillow · matplotlib · Hydra · Weights & Biases

## License

MIT
