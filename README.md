# Mammography Image Quality & Neural Network Robustness

Quantitative analysis of how image quality degradation affects deep learning model performance in mammography screening.

> **MSc Thesis** · Biomedical Engineering, TU Berlin (2026)  
> In cooperation with the Physikalisch-Technische Bundesanstalt (PTB)

## Overview
A reproducible pipeline for image quality metric analysis in mammography, used to quantify how
defined image quality dimensions affect the discrimination, calibration and threshold behaviour of
deep learning classifiers.

Per image:

1. Load the full-resolution DICOM from VinDr-Mammo
2. Compute a breast mask and crop to the breast region
3. Fix the normalisation window and the ROIs (homogeneous tissue patch, lesion box, background
   annulus) on the **original** image
4. Apply the parametrised degradations to the normalised image
5. Compute the image quality metrics on every variant using the **same fixed ROIs**

Step 3 is the design point that makes the comparison fair: regions are determined once on the
undegraded image and reused unchanged for all 31 variants, so a metric change reflects the
degradation and not ROI drift.

**Scope of this repository.** It contains the image quality metric and degradation toolkit. The
degradation parameters below are the ones used for the thesis experiments.

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
Uniform horizontal line PSF simulating patient or paddle motion during exposure, anchored on its
centre pixel, so it is symmetric and preserves the lesion centroid.

The kernel length is specified **directly in pixels**. An earlier mm-first schedule, converted via
`PixelSpacing` with nearest-odd rounding, was abandoned: at the sub-0.1 mm pixel spacings of this
dataset the milder nominal lengths all collapse onto a 3 px kernel and produce identical images.
Fixing the length in pixels guarantees that every severity level is distinct and monotonic.

Kernels apply to the 1024x384 working canvas. One canvas pixel spans a median of 0.240 mm
(IQR 0.213 to 0.281) across the 20,000 images, about 2.8 times the 0.085 mm detector pitch, so the
extents below are median values in the patient plane rather than exact per-image lengths.

| Severity | `length_px` | median extent (mm) |
|---|---|---|
| 1 | 3 | 0.72 |
| 2 | 5 | 1.20 |
| 3 | 7 | 1.68 |
| 4 | 11 | 2.64 |
| 5 | 15 | 3.60 |
| 6 | 21 | 5.04 |

### Contrast compression
Linear contrast compression about the breast-masked median:
`x_out = α·(x_in − c_B) + c_B`, with `c_B` the median over the breast mask. Applied inside the
mask only.

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
Downscale with `INTER_AREA` (area averaging, which avoids aliasing) then upscale back to the
original size with `INTER_CUBIC`, simulating reduced detector resolution or pixel binning.

| Severity | `scale_factor` |
|---|---|
| 1 | 0.90 |
| 2 | 0.80 |
| 3 | 0.70 |
| 4 | 0.60 |
| 5 | 0.50 |
| 6 | 0.40 |

Each batch run produces **31 variants per image**: 1 baseline + 6 noise + 6 motion blur + 6 contrast + 6 JPEG 2000 + 6 resolution.

## Metrics

Computed by this toolkit:

| Metric | Quality dimension | Reference | Scope |
|---|---|---|---|
| `noise_variance` | Noise | No-reference | Homogeneous tissue patch (96x96) |
| `tenengrad` | Sharpness | No-reference | Breast mask |
| `ssim` | Structural integrity | Full-reference | Breast mask |
| `cnr` | Contrast discriminability | No-reference | Lesion vs. background annulus |
| `delta_mu` | Local contrast | No-reference | Lesion vs. background annulus |

`cnr` and `delta_mu` require a finding bounding box and are therefore computed only for annotated
images. `cnr` is `|lesion_mean - bg_mean| / bg_std`; `delta_mu` is `|lesion_mean - bg_mean|`. The
pair is reported together because they share a numerator but differ in denominator, which separates
contrast attenuation (moves `delta_mu`) from raised background noise (moves `cnr` through
`bg_std`). `cnr` is invariant under a global linear contrast scaling, by construction.

The thesis reports **seven** metrics: these five plus the nonparametric noise measure tau and the
robust global contrast (interquartile range over the breast mask), both computed in the evaluation
pipeline rather than in this toolkit.

ROIs are always determined on the **original image** and reused for every degraded variant, so
metric changes are attributable to the degradation.

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
- `metrics_batch.csv`: one row per image × degradation variant
- `run_summary.json`: config, per-image summaries, error log
- `<image_id>/`: QC overlay PNGs and per-image `_summary.json`

## Tech Stack

Python · pydicom · OpenCV · scikit-image · NumPy · SciPy · Pillow · matplotlib

## Citation

If you use this code, please cite both the software and the thesis (see `CITATION.cff`):

> Lorch, N. (2026). *Quantitative Assessment of Image Quality Degradation and Its Impact on the
> Robustness of a Deep Learning Classifier in Mammography*. Master's thesis, Technische
> Universitaet Berlin, in cooperation with the Physikalisch-Technische Bundesanstalt (PTB).

## License

MIT
