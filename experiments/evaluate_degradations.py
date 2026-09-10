#!/usr/bin/env python3
"""
evaluate_degradations.py
========================
Degradation robustness evaluation for the mammography thesis.

For every test image, applies 31 conditions (1 clean + 5 degradation types ×
6 severity levels), runs model inference, computes IQMs, and writes one row
per condition to a CSV.

Usage
-----
    python evaluate_degradations.py \
        --checkpoint  /path/to/ckpt.pth \
        --shards-dir  /path/to/vindr_tar_shards_1024x384_45positive \
        --splits-csv  /path/to/master_splits_1024x384_5positive.csv \
        --arch        convnext_tiny \
        --output      results/convnext_degradation_results.csv

Notes on splits-csv
-------------------
Use master_splits_1024x384_5positive.csv (not the 1024x832 version).
It carries 'breast_birads' (needed to derive both label_45 and label_5)
and 'orig_height' (needed to compute effective pixel spacing after the
tight crop done by dicom_to_tar_shards.py).
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import math
import os
import sys
import tarfile
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import torch
import torch.nn as nn
from skimage.metrics import structural_similarity
from sklearn.metrics import average_precision_score, roc_auc_score
from torchvision import models
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    EfficientNet_B4_Weights,
    ResNet18_Weights,
)
from tqdm import tqdm

import degradations as deg


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEGRADATION_CONDITIONS: list[tuple[str, int]] = [("clean", 0)] + [
    (d, s)
    for d in ["noise", "blur", "contrast", "jpeg2000", "resolution"]
    for s in [1, 2, 3, 4, 5, 6]
]  # 31 total

OUTPUT_COLS = [
    "image_id", "label_45", "label_5", "density", "manufacturer",
    "degradation_type", "severity",
    "logit", "prob",
    "cnr", "delta_mu", "noise_var", "tenengrad", "ssim", "tau",
]

_NAN = float("nan")


# ---------------------------------------------------------------------------
# Glymur smoke test
# ---------------------------------------------------------------------------

def check_glymur() -> bool:
    """Return True if glymur can write+read a small JP2 file."""
    try:
        import glymur  # type: ignore
        test_u16 = (np.ones((64, 64), dtype=np.float32) * 0.5 * 65535).astype(np.uint16)
        tmp = tempfile.mktemp(suffix=".jp2")  # mktemp: no empty file pre-created
        try:
            glymur.Jp2k(tmp, data=test_u16, cratios=[10])
            _ = glymur.Jp2k(tmp)[:]
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        return True
    except Exception as exc:
        print(
            f"WARNING: glymur smoke test failed ({exc}). "
            "JPEG2000 will fall back to JPEG. "
            "Run: pip install --upgrade glymur"
        )
        return False


# ---------------------------------------------------------------------------
# Model construction  (exact match to mammo-18-v3.py / build_model)
# ---------------------------------------------------------------------------

def build_model(arch: str, dropout: float = 0.3) -> nn.Module:
    """Build backbone with the same head structure used during training.

    Uses weights=None — the checkpoint is loaded separately.
    """
    if arch == "resnet18":
        model = models.resnet18(weights=None)
        head_in = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(head_in, 1))

    elif arch == "efficientnet_b4":
        model = models.efficientnet_b4(weights=None)
        head_in = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(head_in, 1)
        )

    elif arch == "convnext_tiny":
        # ConvNeXt head: LayerNorm2d → Flatten → [Dropout] → Linear(768,1)
        # Must preserve the LayerNorm + Flatten layers that precede the FC.
        model = models.convnext_tiny(weights=None)
        head_in = model.classifier[-1].in_features
        new_layers = list(model.classifier.children())[:-1]  # keep everything except final Linear
        new_layers.append(nn.Dropout(dropout))
        new_layers.append(nn.Linear(head_in, 1))
        model.classifier = nn.Sequential(*new_layers)

    else:
        raise ValueError(f"Unknown arch {arch!r}. Choose: resnet18, efficientnet_b4, convnext_tiny")

    return model


def load_checkpoint(path: str, model: nn.Module) -> None:
    """Load state dict, handling the _orig_mod. prefix from torch.compile."""
    ckpt = torch.load(path, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k[len("_orig_mod."):]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)


# ---------------------------------------------------------------------------
# Preprocessing  (must match TarShardDataset.__getitem__ exactly)
# ---------------------------------------------------------------------------

def preprocess(img_np: np.ndarray) -> torch.Tensor:
    """float32 (H,W) [0,1] → (1,3,H,W) float32 tensor."""
    t = torch.from_numpy(img_np).unsqueeze(0).float().repeat(3, 1, 1)
    return t.unsqueeze(0)


# ---------------------------------------------------------------------------
# Per-image setup: tissue patch + sigma0 in one EDT pass
# ---------------------------------------------------------------------------

def _setup_per_image(
    img: np.ndarray,
    mask: np.ndarray,
    patch_size: int = 96,
    contour_margin_px: int = 80,
) -> tuple[float, tuple[int, int, int, int] | None]:
    """Return (sigma0, patch_bbox) from a single EDT pass on the clean image.

    sigma0 is the std of the tissue patch (used for dose_noise).
    patch_bbox is (x1,y1,x2,y2) fixed on the clean image (used for noise_var IQM).
    Falls back to whole-mask std / None if no valid patch is found.
    """
    h, w = img.shape
    step = max(8, patch_size // 4)

    x_g = ndi.gaussian_filter(img.astype(np.float32), sigma=1.0)
    grad = np.hypot(ndi.sobel(x_g, axis=1), ndi.sobel(x_g, axis=0))

    padded = np.pad(mask.astype(bool), pad_width=1, constant_values=False)
    dist = ndi.distance_transform_edt(padded)[1:-1, 1:-1]

    for margin in sorted(
        {contour_margin_px, max(0, contour_margin_px // 2),
         max(0, contour_margin_px // 4), 0},
        reverse=True,
    ):
        candidate = dist >= margin
        if candidate.sum() < patch_size * patch_size:
            continue
        best_score = np.inf
        best_bbox: tuple[int, int, int, int] | None = None
        for y1 in range(0, h - patch_size + 1, step):
            for x1 in range(0, w - patch_size + 1, step):
                y2, x2 = y1 + patch_size, x1 + patch_size
                if candidate[y1:y2, x1:x2].mean() < 1.0:
                    continue
                score = float(grad[y1:y2, x1:x2].mean())
                if score < best_score:
                    best_score = score
                    best_bbox = (x1, y1, x2, y2)
        if best_bbox is not None:
            x1, y1, x2, y2 = best_bbox
            sigma0 = float(np.std(img[y1:y2, x1:x2].astype(np.float32)))
            return sigma0, best_bbox

    sigma0 = float(np.std(img[mask.astype(bool)].astype(np.float32)))
    return sigma0, None


# ---------------------------------------------------------------------------
# Lesion ROI / ring / CNR / delta_mu
# (ported from roi_metrics_dicom_3.0.py — bbox_mask, lesion_background_ring,
#  compute_cnr — kept self-contained here to avoid a circular import; the
#  geometry must match exactly since lesion bboxes are precomputed in canvas
#  coordinates by build_lesion_roi_lookup.py and reused as FIXED ROIs)
# ---------------------------------------------------------------------------

_CNR_EPS = 1e-8

BBoxT = tuple[int, int, int, int]


def bbox_mask(shape: tuple[int, int], bbox: BBoxT) -> np.ndarray:
    h, w = shape
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(w, int(x1)))
    x2 = max(0, min(w, int(x2)))
    y1 = max(0, min(h, int(y1)))
    y2 = max(0, min(h, int(y2)))
    m = np.zeros((h, w), dtype=bool)
    if x2 > x1 and y2 > y1:
        m[y1:y2, x1:x2] = True
    return m


def lesion_background_ring(
    lesion_bbox: BBoxT,
    image_shape: tuple[int, int],
    breast_mask: np.ndarray,
    other_lesions: list[BBoxT] | None = None,
    inner_scale: float = 1.5,
    outer_scale: float = 2.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Lesion mask + surrounding ring/annulus background mask.

    Mirrors roi_metrics_dicom_3.0.lesion_background_ring: ring = (outer box
    minus inner box) intersected with the breast mask, with other annotated
    lesions excluded so the background estimate isn't contaminated.
    """
    h, w = image_shape
    x1, y1, x2, y2 = lesion_bbox
    cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    bw, bh = x2 - x1, y2 - y1

    def scaled_box(scale: float) -> BBoxT:
        nw = max(1, int(round(bw * scale)))
        nh = max(1, int(round(bh * scale)))
        sx1 = int(round(cx - nw / 2))
        sy1 = int(round(cy - nh / 2))
        sx2 = int(round(cx + nw / 2))
        sy2 = int(round(cy + nh / 2))
        return (max(0, min(w, sx1)), max(0, min(h, sy1)),
                max(0, min(w, sx2)), max(0, min(h, sy2)))

    inner_m = bbox_mask((h, w), scaled_box(inner_scale))
    outer_m = bbox_mask((h, w), scaled_box(outer_scale))
    lesion_m = bbox_mask((h, w), lesion_bbox)
    ring = outer_m & (~inner_m) & breast_mask.astype(bool)

    if other_lesions:
        other_mask = np.zeros((h, w), dtype=bool)
        for b in other_lesions:
            if tuple(b) != tuple(lesion_bbox):
                other_mask |= bbox_mask((h, w), tuple(b))
        ring &= ~other_mask

    return lesion_m, ring


def lesion_cnr_delta_mu(
    img: np.ndarray,
    lesion_mask: np.ndarray,
    ring_mask: np.ndarray,
) -> tuple[float, float]:
    """delta_mu = |mean_lesion - mean_ring|; cnr = delta_mu / (std_ring + eps).

    Per the thesis definition (matches the formula specified for this
    integration — note this differs slightly from roi_metrics_dicom_3.0's
    compute_cnr, which returns +inf for a zero-variance background instead
    of using an epsilon; the epsilon form is more stable for downstream
    correlation/statistics).
    """
    lesion_vals = np.asarray(img[lesion_mask], dtype=np.float32)
    bg_vals = np.asarray(img[ring_mask], dtype=np.float32)
    if lesion_vals.size == 0 or bg_vals.size == 0:
        return _NAN, _NAN
    mean_lesion = float(np.mean(lesion_vals))
    mean_ring = float(np.mean(bg_vals))
    std_ring = float(np.std(bg_vals, ddof=0))
    delta_mu = abs(mean_lesion - mean_ring)
    cnr = delta_mu / (std_ring + _CNR_EPS)
    return cnr, delta_mu


# ---------------------------------------------------------------------------
# tau — nonparametric noise measure (Anton et al. 2023)
# ---------------------------------------------------------------------------

# Anton, Mäder, Schopphoven, Reginatto, "A Nonparametric Measure of Noise in
# X-Ray Diagnostic Images — Mammography", Phys. Med. Biol. 68(4):045003 (2023),
# doi:10.1088/1361-6560/acb485. Builds on the non-maximum-suppression (NMS)
# statistic of Obuchowicz et al. (Entropy 22(2):220, 2020).
_TAU_N_THRESHOLDS = 2000  # paper recommends N_t >= 1000; ~2000 for clinical use


def compute_tau(
    img: np.ndarray,
    mask: np.ndarray | None = None,
    n_thresholds: int = _TAU_N_THRESHOLDS,
) -> float:
    """Nonparametric noise measure tau (Anton et al. 2023, eq. 2).

    For every interior pixel p, let d(p) be the MINIMUM over its 8 nearest
    neighbours of |I(p) - I(neighbour)|. The NMS criterion counts a pixel at
    threshold t iff the gray-value difference to *all eight* neighbours exceeds
    t, i.e. iff d(p) > t, so

        n(t) = #{ pixels : d(p) > t }

    a monotonically decreasing function of t. tau is the n(t)-weighted mean
    threshold:

        tau = sum_i t_i * n(t_i) / sum_i n(t_i)

    over a uniform grid t_i = i * dt, i = 1..n_thresholds, dt = t_max / n_thresholds,
    t_max = max(I) - min(I) on the evaluated region. tau is proportional to the
    noise standard deviation and is insensitive to large-scale anatomy.

    Adaptation for this pipeline: restricted to the breast mask when given —
    consistent with tenengrad/noise_var and to exclude the zero-padding border
    of the 1024x384 canvas (whose min=0 would otherwise inflate t_max). Only
    pixels whose full 8-neighbourhood lies inside the mask are used; t_max is
    taken over the masked region. No-reference: computed on the (degraded) image
    alone.
    """
    x = img.astype(np.float32)
    # |difference| to each of the 8 neighbours, for interior pixels only
    c = x[1:-1, 1:-1]
    diffs = np.stack([
        np.abs(c - x[:-2,  :-2]), np.abs(c - x[:-2, 1:-1]), np.abs(c - x[:-2, 2:]),
        np.abs(c - x[1:-1, :-2]),                           np.abs(c - x[1:-1, 2:]),
        np.abs(c - x[2:,   :-2]), np.abs(c - x[2:,  1:-1]), np.abs(c - x[2:,  2:]),
    ], axis=0)
    d = diffs.min(axis=0)  # NMS: counted iff diff to ALL neighbours > t  <=>  min > t

    if mask is not None:
        m = mask.astype(bool)
        # require the full 8-neighbourhood to lie inside the breast mask
        interior = (
            m[1:-1, 1:-1] & m[:-2, :-2] & m[:-2, 1:-1] & m[:-2, 2:] &
            m[1:-1, :-2] & m[1:-1, 2:] &
            m[2:, :-2] & m[2:, 1:-1] & m[2:, 2:]
        )
        d = d[interior]
        region = x[m]
    else:
        d = d.ravel()
        region = x.ravel()

    if d.size == 0 or region.size == 0:
        return _NAN
    t_max = float(region.max() - region.min())
    if t_max <= 0:
        return 0.0

    dt = t_max / n_thresholds
    t_grid = np.arange(1, n_thresholds + 1, dtype=np.float64) * dt
    d_sorted = np.sort(d.astype(np.float64))
    # n(t_i) = #{ d > t_i } = N - (#{ d <= t_i })
    n_t = d_sorted.size - np.searchsorted(d_sorted, t_grid, side="right")
    denom = float(n_t.sum())
    if denom == 0.0:
        return 0.0
    return float(np.sum(t_grid * n_t) / denom)


# ---------------------------------------------------------------------------
# IQM computation
# ---------------------------------------------------------------------------

def compute_iqms(
    img_deg: np.ndarray,
    img_clean: np.ndarray,
    mask: np.ndarray,
    patch_bbox: tuple[int, int, int, int] | None,
    lesion_mask: np.ndarray | None = None,
    ring_mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute cnr, delta_mu, noise_var, tenengrad, ssim, tau from fixed ROIs.

    lesion_mask / ring_mask are FIXED ROIs determined once on the clean
    image (via lesion_background_ring, using the precomputed lesion bbox
    from build_lesion_roi_lookup.py) and reused for every degraded variant —
    same "fixed ROI" principle as patch_bbox and the breast mask. They are
    None for images without lesion annotations, in which case cnr/delta_mu
    remain NaN.
    """
    result: dict[str, float] = {}

    # cnr / delta_mu: lesion vs. surrounding ring, on fixed ROIs
    if lesion_mask is not None and ring_mask is not None:
        try:
            result["cnr"], result["delta_mu"] = lesion_cnr_delta_mu(img_deg, lesion_mask, ring_mask)
        except Exception as exc:
            warnings.warn(f"cnr/delta_mu failed: {exc}")
            result["cnr"], result["delta_mu"] = _NAN, _NAN
    else:
        result["cnr"], result["delta_mu"] = _NAN, _NAN

    # noise_var: variance of tissue patch on degraded image
    try:
        if patch_bbox is not None:
            x1, y1, x2, y2 = patch_bbox
            patch = img_deg[y1:y2, x1:x2].astype(np.float32)
            result["noise_var"] = float(np.var(patch))
        else:
            result["noise_var"] = _NAN
    except Exception as exc:
        warnings.warn(f"noise_var failed: {exc}")
        result["noise_var"] = _NAN

    # tenengrad: mean squared Sobel gradient over breast mask
    try:
        x = img_deg.astype(np.float32)
        g2 = ndi.sobel(x, axis=1) ** 2 + ndi.sobel(x, axis=0) ** 2
        result["tenengrad"] = float(np.mean(g2[mask.astype(bool)]))
    except Exception as exc:
        warnings.warn(f"tenengrad failed: {exc}")
        result["tenengrad"] = _NAN

    # ssim: full-reference vs clean image
    try:
        result["ssim"] = float(
            structural_similarity(
                img_clean.astype(np.float32),
                img_deg.astype(np.float32),
                data_range=1.0,
            )
        )
    except Exception as exc:
        warnings.warn(f"ssim failed: {exc}")
        result["ssim"] = _NAN

    # tau: nonparametric noise measure (Anton et al. 2023), no-reference,
    # restricted to the breast mask
    try:
        result["tau"] = compute_tau(img_deg, mask=mask)
    except Exception as exc:
        warnings.warn(f"tau failed: {exc}")
        result["tau"] = _NAN

    return result


# ---------------------------------------------------------------------------
# Effective pixel spacing
# ---------------------------------------------------------------------------

def effective_pixel_spacing(orig_h: int | None, img: np.ndarray) -> float:
    """Estimate pixel spacing (mm) after crop+resize preprocessing.

    orig_ps = 0.07 mm (default for VinDr-Mammo Siemens units).
    H_content = number of non-zero rows in the processed image (breast region).
    effective_ps = orig_ps * (orig_h / H_content).
    """
    H_content = int(np.count_nonzero(img.any(axis=1)))
    if orig_h and H_content > 0:
        return 0.07 * (float(orig_h) / H_content)
    return 0.07


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _fmt(v: float) -> str:
    if isinstance(v, float) and math.isnan(v):
        return ""
    return str(round(v, 6))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Degradation robustness evaluation — writes one CSV row per (image × condition).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint",  type=str, required=True,  help=".pth checkpoint file")
    p.add_argument("--shards-dir",  type=str, required=True,  help="Root shard dir (contains test/, metadata/)")
    p.add_argument(
        "--splits-csv", type=str, required=True,
        help="master_splits_1024x384_5positive.csv — provides breast_birads and orig_height",
    )
    p.add_argument(
        "--lesion-roi-csv", type=str, default=None,
        help="lesion_roi_lookup_1024x384.csv (from build_lesion_roi_lookup.py). "
             "Maps annotated images to a primary lesion bbox + other-lesion "
             "bboxes in 1024x384 CANVAS coordinates, precomputed locally "
             "(the cluster has no DICOM access to derive these on the fly). "
             "If omitted, cnr/delta_mu remain NaN for every image.",
    )
    p.add_argument("--arch",   type=str, default="convnext_tiny",
                   choices=["convnext_tiny", "resnet18", "efficientnet_b4"])
    p.add_argument("--output", type=str, required=True, help="Output CSV path")
    p.add_argument("--batch-size", type=int, default=1,
                   help="Kept for API compatibility; inference is per-image")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--dry-run", action="store_true", help="Process first 50 test images only")
    p.add_argument(
        "--sanity-check", type=str, default=None, metavar="ORIG_TEST_PREDICTIONS_CSV",
        help="Path to mammo-18-v3.py's test_predictions.csv. If given, runs ONLY "
             "the clean condition for every test image, merges by image_id with "
             "the original predictions, prints AUC/AP/correlation/mean&max abs "
             "diff, and exits WITHOUT writing the degradation results CSV. Run "
             "this first — clean AUC should be ≈ the original test AUC, "
             "correlation ≈ 1.0, mean |diff| ≈ 0 — before evaluating degraded "
             "conditions.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Sanity check: clean-condition predictions vs. original test_predictions.csv
# ---------------------------------------------------------------------------

def run_sanity_check(
    args: argparse.Namespace,
    model: nn.Module,
    device: torch.device,
    use_amp: bool,
    test_manifest: pd.DataFrame,
    log: logging.Logger,
) -> None:
    """Run only the clean (undegraded) condition and compare against the
    original mammo-18-v3.py test_predictions.csv.

    This isolates the inference/preprocessing pipeline from the degradation
    machinery: if clean predictions don't match the original run's predictions
    by image_id, the bug is in loading/preprocessing/model/precision — not in
    anything degradation-specific.
    """
    baseline = pd.read_csv(args.sanity_check)
    baseline["image_id"] = baseline["image_id"].astype(str)
    baseline = baseline.set_index("image_id")

    _tar_cache: dict[str, tarfile.TarFile] = {}

    def _get_tar(path: str) -> tarfile.TarFile:
        if path not in _tar_cache:
            _tar_cache[path] = tarfile.open(path, "r:*")
        return _tar_cache[path]

    clean_probs: dict[str, float] = {}
    for _, row in tqdm(test_manifest.iterrows(), total=len(test_manifest),
                       desc="sanity-check (clean only)", ncols=90):
        image_id   = str(row["image_id"])
        shard_path = str(Path(args.shards_dir) / "test" / str(row["shard"]))
        try:
            tf     = _get_tar(shard_path)
            img_np = np.load(io.BytesIO(tf.extractfile(f"{image_id}.npy").read())).astype(np.float32)
        except Exception as exc:
            log.warning(f"sanity-check: cannot load {image_id}: {exc} — skipping")
            continue

        t = preprocess(img_np).float().to(device)
        with torch.no_grad():
            if use_amp:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    logits = model(t).view(-1)
            else:
                logits = model(t).view(-1)
            prob = float(torch.sigmoid(logits).float().cpu().item())
        clean_probs[image_id] = prob

    for tf in _tar_cache.values():
        try:
            tf.close()
        except Exception:
            pass

    clean_df = pd.DataFrame(
        {"image_id": list(clean_probs.keys()), "clean_prob": list(clean_probs.values())}
    ).set_index("image_id")
    merged = baseline.join(clean_df, how="inner")

    if merged.empty or "pred_prob" not in merged.columns or "true_label" not in merged.columns:
        log.warning(
            "sanity-check: no usable overlap with baseline CSV — need "
            "'image_id', 'true_label', 'pred_prob' columns (the file produced "
            "by mammo-18-v3.py as test_predictions.csv)."
        )
        return

    y_true  = merged["true_label"].to_numpy()
    p_orig  = merged["pred_prob"].to_numpy()
    p_clean = merged["clean_prob"].to_numpy()

    auc_orig  = roc_auc_score(y_true, p_orig)
    auc_clean = roc_auc_score(y_true, p_clean)
    ap_orig   = average_precision_score(y_true, p_orig)
    ap_clean  = average_precision_score(y_true, p_clean)
    corr      = float(np.corrcoef(p_orig, p_clean)[0, 1])
    mad       = float(np.mean(np.abs(p_orig - p_clean)))
    maxad     = float(np.max(np.abs(p_orig - p_clean)))

    log.info(
        f"\n{'='*60}\n"
        f"  SANITY CHECK — clean condition vs. {Path(args.sanity_check).name}\n"
        f"{'='*60}\n"
        f"  matched images           : {len(merged)} / {len(baseline)}\n"
        f"  AUC   original vs clean  : {auc_orig:.10f}  vs  {auc_clean:.10f}\n"
        f"  AP    original vs clean  : {ap_orig:.10f}  vs  {ap_clean:.10f}\n"
        f"  correlation              : {corr:.6f}\n"
        f"  mean absolute difference : {mad:.6f}\n"
        f"  max absolute difference  : {maxad:.6f}\n"
        f"{'='*60}\n"
        f"  Goal: clean AUC ≈ {auc_orig:.4f}, correlation ≈ 1.0, mean|Δ| ≈ 0\n"
        f"{'='*60}"
    )


# ---------------------------------------------------------------------------
# Diagnostic summary: cnr / delta_mu coverage
# ---------------------------------------------------------------------------

def print_lesion_roi_diagnostics(out_path: Path, log: logging.Logger) -> None:
    """Print non-null counts for cnr/delta_mu — overall, by degradation_type,
    and by label_45 — by reading back the (possibly resumed/appended) output CSV.

    This reports on the FULL output file, not just rows written this run, so
    it remains accurate across crash-recovery resumes.
    """
    try:
        df = pd.read_csv(out_path)
    except Exception as exc:
        log.warning(f"Could not read {out_path} for diagnostics: {exc}")
        return

    n_total = len(df)
    n_cnr   = int(df["cnr"].notna().sum())
    n_dmu   = int(df["delta_mu"].notna().sum())

    by_type = df.groupby("degradation_type")[["cnr", "delta_mu"]].apply(
        lambda g: pd.Series({"cnr": int(g["cnr"].notna().sum()),
                             "delta_mu": int(g["delta_mu"].notna().sum())})
    )
    by_label = df.groupby("label_45")[["cnr", "delta_mu"]].apply(
        lambda g: pd.Series({"cnr": int(g["cnr"].notna().sum()),
                             "delta_mu": int(g["delta_mu"].notna().sum())})
    )

    lines = [
        f"\n{'='*55}",
        "  LESION ROI DIAGNOSTIC SUMMARY (cnr / delta_mu)",
        f"{'='*55}",
        f"  non-null cnr        : {n_cnr} / {n_total}",
        f"  non-null delta_mu   : {n_dmu} / {n_total}",
        "",
        "  by degradation_type (non-null counts):",
        *(f"    {dtype:<12s} cnr={int(r['cnr']):>6d}   delta_mu={int(r['delta_mu']):>6d}"
          for dtype, r in by_type.iterrows()),
        "",
        "  by label_45 (non-null counts):",
        *(f"    label_45={lbl:<3} cnr={int(r['cnr']):>6d}   delta_mu={int(r['delta_mu']):>6d}"
          for lbl, r in by_label.iterrows()),
        f"{'='*55}",
    ]
    log.info("\n".join(lines))


# ---------------------------------------------------------------------------
# Dry-run sanity checks
# ---------------------------------------------------------------------------

def check_lesion_roi_nonnull(records: list[dict], log: logging.Logger) -> None:
    """Confirm cnr/delta_mu are no longer all-NaN for annotated positive images."""
    pos_annotated = [r for r in records if r["label_45"] == 1 and r["has_lesion_roi"]]
    n = len(pos_annotated)
    n_cnr = sum(1 for r in pos_annotated if not (isinstance(r["cnr"], float) and math.isnan(r["cnr"])))
    n_dmu = sum(1 for r in pos_annotated if not (isinstance(r["delta_mu"], float) and math.isnan(r["delta_mu"])))

    lines = [
        f"\n{'='*55}",
        "  CHECK — cnr/delta_mu for annotated POSITIVE images (dry-run)",
        f"{'='*55}",
        f"  rows (label_45=1 & has lesion ROI): {n}",
        f"  non-null cnr      : {n_cnr} / {n}",
        f"  non-null delta_mu : {n_dmu} / {n}",
        f"  Result: {'PASS — no longer all NaN' if (n > 0 and n_cnr > 0 and n_dmu > 0) else 'FAIL — still all NaN (or no annotated positives in this sample)'}",
        f"{'='*55}",
    ]
    log.info("\n".join(lines))


def check_blur_severity_distinct(records: list[dict], log: logging.Logger) -> None:
    """Confirm blur severities 1-6 produce distinct predictions (identical=False)."""
    from collections import defaultdict

    probs: dict[str, dict[int, float]] = defaultdict(dict)
    for r in records:
        if r["degradation_type"] == "blur":
            probs[r["image_id"]][r["severity"]] = r["prob"]

    severities = [1, 2, 3, 4, 5, 6]
    lines = [f"\n{'='*55}", "  CHECK — blur severity distinctness (dry-run)", f"{'='*55}"]
    all_distinct = True
    any_pair = False
    for i, s1 in enumerate(severities):
        for s2 in severities[i + 1:]:
            diffs = [abs(probs[iid][s1] - probs[iid][s2])
                     for iid in probs if s1 in probs[iid] and s2 in probs[iid]]
            if not diffs:
                continue
            any_pair = True
            identical = all(d == 0.0 for d in diffs)
            all_distinct = all_distinct and not identical
            lines.append(
                f"    s{s1} vs s{s2}: identical={identical!s:<5}  "
                f"mean|Δprob|={float(np.mean(diffs)):.6f}  max|Δprob|={float(np.max(diffs)):.6f}"
            )
    result = "PASS — all pairs identical=False" if (any_pair and all_distinct) else "FAIL — some severities identical"
    lines.append(f"  Result: {result}")
    lines.append(f"{'='*55}")
    log.info("\n".join(lines))


def main() -> None:
    args = parse_args()
    t_start = time.time()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = out_path.with_suffix(".log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(log_path), mode="a"),
        ],
    )
    log = logging.getLogger(__name__)

    # ── Glymur check ──────────────────────────────────────────────────────────
    GLYMUR_OK = check_glymur()
    log.info(f"GLYMUR_OK: {GLYMUR_OK}")

    # ── Device ────────────────────────────────────────────────────────────────
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    log.info(f"Device: {device}")

    # ── Model ─────────────────────────────────────────────────────────────────
    # IMPORTANT: must mirror mammo-18-v3.py's evaluate_test() exactly. There,
    # build_model() returns model.to(device) WITHOUT .half() — weights stay
    # float32, and torch.cuda.amp.autocast(dtype=torch.float16) provides mixed
    # precision around the forward pass only. Calling model.half() (as this
    # script previously did) permanently casts ALL weights — including
    # LayerNorm gamma/beta that ConvNeXt relies on heavily — to fp16, which is
    # numerically very different from autocast's mixed-precision policy (which
    # keeps reduction-sensitive ops like LayerNorm in fp32). That mismatch is
    # what broke reproduction of the original test_predictions.csv.
    model = build_model(args.arch, dropout=0.3)
    load_checkpoint(args.checkpoint, model)
    model.eval()
    model = model.float().to(device)
    use_amp = device.type == "cuda"
    log.info(f"Checkpoint loaded: {args.checkpoint}  device={device}  amp={use_amp}")

    # ── Splits metadata ───────────────────────────────────────────────────────
    splits_df = pd.read_csv(args.splits_csv)
    if "breast_birads" in splits_df.columns:
        splits_df["label_45"] = (splits_df["breast_birads"] >= 4).astype(int)
        splits_df["label_5"]  = (splits_df["breast_birads"] >= 5).astype(int)
    else:
        log.warning(
            "splits-csv has no 'breast_birads' column — both label_45 and label_5 "
            "will use the single 'label' column. Pass master_splits_1024x384_5positive.csv."
        )
        splits_df["label_45"] = splits_df["label"]
        splits_df["label_5"]  = splits_df["label"]
    splits_idx = splits_df.set_index("image_id")

    # ── Lesion ROI lookup (precomputed canvas-coordinate bboxes) ─────────────
    lesion_lookup: dict[str, dict] = {}
    if args.lesion_roi_csv:
        lroi = pd.read_csv(args.lesion_roi_csv)
        lroi["image_id"] = lroi["image_id"].astype(str)
        for _, r in lroi.iterrows():
            try:
                other = json.loads(r["other_lesion_bboxes"]) if isinstance(r["other_lesion_bboxes"], str) else []
            except (TypeError, ValueError, json.JSONDecodeError):
                other = []
            lesion_lookup[str(r["image_id"])] = {
                "bbox": (int(r["lesion_xmin"]), int(r["lesion_ymin"]),
                         int(r["lesion_xmax"]), int(r["lesion_ymax"])),
                "other": [tuple(b) for b in other],
            }
        log.info(f"Loaded lesion ROI lookup: {len(lesion_lookup)} annotated images "
                 f"(from {args.lesion_roi_csv})")
    else:
        log.info("No --lesion-roi-csv given — cnr/delta_mu will remain NaN for every image")

    # ── Shard manifest ────────────────────────────────────────────────────────
    manifest_path = Path(args.shards_dir) / "metadata" / "shard_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    test_manifest = manifest[manifest["split"] == "test"].reset_index(drop=True)

    if args.dry_run:
        test_manifest = test_manifest.head(50)
        log.info("dry-run: capped at 50 test images")

    n_test = len(test_manifest)

    # ── Sanity check mode (clean condition only, then exit) ──────────────────
    if args.sanity_check:
        log.info(f"Sanity-check mode: comparing clean predictions against {args.sanity_check}")
        run_sanity_check(args, model, device, use_amp, test_manifest, log)
        return

    # ── Crash recovery ────────────────────────────────────────────────────────
    done_ids: set[str] = set()
    write_header = not out_path.exists() or out_path.stat().st_size == 0
    if not write_header:
        try:
            existing = pd.read_csv(out_path, usecols=["image_id"])
            counts = existing["image_id"].value_counts()
            done_ids = set(counts[counts >= 31].index.astype(str))
        except Exception as exc:
            log.warning(f"Could not read existing output for resume: {exc}")

    todo = test_manifest[~test_manifest["image_id"].astype(str).isin(done_ids)]

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info(
        f"\n{'='*55}\n"
        f"  Checkpoint:   {args.checkpoint}\n"
        f"  Architecture: {args.arch}\n"
        f"  Test images:  {n_test}{' (dry-run)' if args.dry_run else ''}\n"
        f"  Conditions:   31 (1 clean + 5×6 degraded)\n"
        f"  Output:       {out_path}\n"
        f"  Already done: {len(done_ids)} images (resuming)\n"
        f"  Remaining:    {len(todo)} images\n"
        f"{'='*55}"
    )

    if len(todo) == 0:
        log.info("Nothing to do — all images already processed.")
        return

    # ── Open output CSV (append mode) ─────────────────────────────────────────
    with open(out_path, "a", newline="", encoding="utf-8") as f_out:
        writer = csv.writer(f_out)
        if write_header:
            writer.writerow(OUTPUT_COLS)

        total_rows = 0
        dry_run_records: list[dict] = []  # only populated when --dry-run (small, in-memory)

        # ── Tar handle cache (one open handle per shard file) ─────────────────
        _tar_cache: dict[str, tarfile.TarFile] = {}

        def _get_tar(path: str) -> tarfile.TarFile:
            if path not in _tar_cache:
                _tar_cache[path] = tarfile.open(path, "r:*")
            return _tar_cache[path]

        # ── Main loop ─────────────────────────────────────────────────────────
        for img_idx, (_, row) in enumerate(
            tqdm(todo.iterrows(), total=len(todo), desc="images", ncols=90)
        ):
            image_id  = str(row["image_id"])
            shard_path = str(Path(args.shards_dir) / "test" / str(row["shard"]))

            # Load image and JSON sidecar from TAR shard
            try:
                tf = _get_tar(shard_path)
                img_np = np.load(io.BytesIO(tf.extractfile(f"{image_id}.npy").read()))
                sidecar = json.loads(tf.extractfile(f"{image_id}.json").read())
            except Exception as exc:
                log.warning(f"Cannot load {image_id}: {exc} — skipping")
                continue

            img_np = img_np.astype(np.float32)

            # Metadata from splits CSV (preferred) with sidecar as fallback
            if image_id in splits_idx.index:
                si = splits_idx.loc[image_id]
                label_45     = int(si.get("label_45", sidecar.get("label", 0)))
                label_5      = int(si.get("label_5",  sidecar.get("label", 0)))
                density      = str(si.get("density",      sidecar.get("density",      "")))
                manufacturer = str(si.get("manufacturer", sidecar.get("manufacturer", "")))
                orig_h       = si.get("orig_height", None)
                orig_h       = int(orig_h) if orig_h and not (isinstance(orig_h, float) and math.isnan(orig_h)) else None
            else:
                log.warning(f"{image_id} not found in splits-csv — using sidecar labels only")
                label_45     = int(sidecar.get("label", 0))
                label_5      = int(sidecar.get("label", 0))
                density      = str(sidecar.get("density", ""))
                manufacturer = str(sidecar.get("manufacturer", ""))
                orig_h       = None

            # Per-image setup (computed once on clean image)
            try:
                mask    = deg.get_breast_mask(img_np)
                sigma0, patch_bbox = _setup_per_image(img_np, mask)
                eff_ps  = effective_pixel_spacing(orig_h, img_np)
            except Exception as exc:
                log.warning(f"Per-image setup failed for {image_id}: {exc} — skipping")
                continue

            # Fixed lesion + ring ROI (only for annotated images), determined
            # ONCE on the clean image and reused for every degraded variant —
            # same fixed-ROI principle as patch_bbox/breast mask above.
            lesion_mask = ring_mask = None
            roi_entry = lesion_lookup.get(image_id)
            if roi_entry is not None:
                try:
                    lesion_mask, ring_mask = lesion_background_ring(
                        roi_entry["bbox"], img_np.shape, mask,
                        other_lesions=roi_entry["other"],
                    )
                    if not np.any(lesion_mask) or not np.any(ring_mask):
                        lesion_mask = ring_mask = None
                except Exception as exc:
                    log.warning(f"{image_id}: lesion ROI/ring failed: {exc}")
                    lesion_mask = ring_mask = None

            # ── 31 conditions ──────────────────────────────────────────────
            rows_buffer: list[list] = []

            for deg_type, severity in DEGRADATION_CONDITIONS:
                # Degradation
                if deg_type == "clean":
                    img_deg = img_np
                else:
                    try:
                        img_deg = deg.apply_degradation(
                            img_np, deg_type, severity,
                            mask=mask,
                            sigma0=sigma0,
                            pixel_spacing_mm=eff_ps,
                            seed=img_idx,
                        )
                    except Exception as exc:
                        log.warning(f"{image_id} {deg_type} s={severity} degradation failed: {exc}")
                        img_deg = img_np

                # Inference — mirrors evaluate_test(): fp32 input + fp32 weights,
                # autocast(dtype=float16) around the forward only, sigmoid applied
                # to the raw (possibly fp16) logits outside the autocast block,
                # exactly like `probs = torch.sigmoid(logits).float()...` there.
                try:
                    t = preprocess(img_deg).float().to(device)
                    with torch.no_grad():
                        if use_amp:
                            with torch.cuda.amp.autocast(dtype=torch.float16):
                                logits = model(t).view(-1)
                        else:
                            logits = model(t).view(-1)
                        prob_t = torch.sigmoid(logits).float()
                    logit = float(logits.float().cpu().item())
                    prob  = float(prob_t.cpu().item())
                except Exception as exc:
                    log.warning(f"{image_id} inference failed: {exc}")
                    logit, prob = _NAN, _NAN

                # IQMs
                try:
                    iqms = compute_iqms(img_deg, img_np, mask, patch_bbox,
                                        lesion_mask=lesion_mask, ring_mask=ring_mask)
                except Exception as exc:
                    log.warning(f"{image_id} IQMs failed: {exc}")
                    iqms = {"cnr": _NAN, "delta_mu": _NAN, "noise_var": _NAN,
                            "tenengrad": _NAN, "ssim": _NAN, "tau": _NAN}

                rows_buffer.append([
                    image_id, label_45, label_5, density, manufacturer,
                    deg_type, severity,
                    _fmt(logit), _fmt(prob),
                    _fmt(iqms["cnr"]),   _fmt(iqms["delta_mu"]),
                    _fmt(iqms["noise_var"]), _fmt(iqms["tenengrad"]),
                    _fmt(iqms["ssim"]), _fmt(iqms["tau"]),
                ])

                if args.dry_run:
                    dry_run_records.append({
                        "image_id": image_id, "label_45": label_45,
                        "has_lesion_roi": roi_entry is not None,
                        "degradation_type": deg_type, "severity": severity,
                        "prob": prob, "cnr": iqms["cnr"], "delta_mu": iqms["delta_mu"],
                    })

            # Write all 31 rows for this image at once, then flush
            writer.writerows(rows_buffer)
            f_out.flush()
            total_rows += len(rows_buffer)

        # Close cached tar handles
        for tf in _tar_cache.values():
            try:
                tf.close()
            except Exception:
                pass

    # ── Final summary ─────────────────────────────────────────────────────────
    runtime = time.time() - t_start
    file_mb = out_path.stat().st_size / 1e6
    log.info(
        f"\n{'='*55}\n"
        f"  Total rows written: {total_rows}\n"
        f"  Estimated file size: {file_mb:.1f} MB\n"
        f"  Runtime: {runtime / 60:.1f} min\n"
        f"{'='*55}"
    )

    # ── Lesion ROI diagnostic summary (cnr / delta_mu coverage) ──────────────
    print_lesion_roi_diagnostics(out_path, log)

    # ── Dry-run sanity checks ─────────────────────────────────────────────────
    if args.dry_run:
        check_lesion_roi_nonnull(dry_run_records, log)
        check_blur_severity_distinct(dry_run_records, log)


if __name__ == "__main__":
    main()
