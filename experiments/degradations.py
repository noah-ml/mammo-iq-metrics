#!/usr/bin/env python3
"""
degradations.py
===============
Five degradation functions for the mammography robustness thesis pipeline.

Each function accepts a float32 numpy array of shape (H, W) with values
in [0, 1] and returns a float32 array of the same shape, clipped to [0, 1].

Functions
---------
dose_noise          – Poisson-motivated additive Gaussian noise (eq 4.7–4.9)
motion_blur         – Horizontal (θ=0°) motion blur (eq 4.11)
contrast_reduction  – Breast-masked mean-centering contrast compression (eq 4.12–4.13)
jpeg2000            – JPEG2000 lossy encode-decode at given compression ratio (eq 4.14–4.15)
resolution_reduction– INTER_AREA downscale + INTER_CUBIC upsample (eq 4.16)

Helpers
-------
get_breast_mask     – Otsu-fraction + largest connected component + convex hull
                      (extracted from roi_metrics_dicom_3.0.py; no circular dep)
get_sigma0          – std of EDT-selected homogeneous tissue patch
get_pixel_spacing   – Read pixel spacing from a TAR-shard JSON metadata dict

Convenience
-----------
apply_degradation   – Dispatch all five functions by name
DEGRADATION_TYPES, SEVERITY_LEVELS
"""
from __future__ import annotations

import math
import os
import tempfile
import warnings
from typing import Any

import cv2
import numpy as np
import scipy.ndimage as ndi


# ---------------------------------------------------------------------------
# Internal helpers for get_breast_mask
# (Logic extracted from roi_metrics_dicom_3.0.py; uses only numpy/scipy/cv2)
# ---------------------------------------------------------------------------

def _normalize_for_mask(img: np.ndarray) -> np.ndarray:
    x = np.asarray(img, dtype=np.float32)
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x)
    vals = x[finite]
    mn, mx = float(vals.min()), float(vals.max())
    if mx <= mn:
        return np.zeros_like(x)
    out = (x - mn) / (mx - mn)
    out[~finite] = 0.0
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _otsu_threshold(vals: np.ndarray) -> float:
    vals = vals.astype(np.float64)
    mn, mx = float(vals.min()), float(vals.max())
    if mx <= mn:
        return float(mn)
    hist, bin_edges = np.histogram(vals, bins=256, range=(mn, mx))
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    w = hist.astype(np.float64) / hist.sum()
    w0 = np.cumsum(w)
    w1 = 1.0 - w0
    mu0 = np.cumsum(w * bin_centers) / np.maximum(w0, 1e-12)
    mu1 = (np.sum(w * bin_centers) - np.cumsum(w * bin_centers)) / np.maximum(w1, 1e-12)
    sigma_b2 = w0 * w1 * (mu0 - mu1) ** 2
    return float(bin_centers[np.argmax(sigma_b2)])


def _largest_component(mask: np.ndarray) -> np.ndarray:
    labeled, n = ndi.label(mask.astype(bool))
    if n == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = ndi.sum(np.ones_like(labeled, dtype=np.int32), labeled, range(1, n + 1))
    return (labeled == (int(np.argmax(sizes)) + 1))


def _remove_small_objects(mask: np.ndarray, min_size: int) -> np.ndarray:
    labeled, n = ndi.label(mask.astype(bool))
    if n == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = ndi.sum(np.ones_like(labeled, dtype=np.int32), labeled, range(1, n + 1))
    out = np.zeros_like(mask, dtype=bool)
    for label_idx, sz in enumerate(sizes, start=1):
        if sz >= min_size:
            out |= (labeled == label_idx)
    return out


def _remove_small_holes(mask: np.ndarray, area_threshold: int) -> np.ndarray:
    inv = ~mask.astype(bool)
    labeled, n = ndi.label(inv)
    if n == 0:
        return mask.astype(bool)
    out = mask.astype(bool).copy()
    sizes = ndi.sum(np.ones_like(labeled, dtype=np.int32), labeled, range(1, n + 1))
    for label_idx, sz in enumerate(sizes, start=1):
        if sz <= area_threshold:
            out |= (labeled == label_idx)
    return out


def _convex_hull_image(mask: np.ndarray) -> np.ndarray:
    try:
        from skimage.morphology import convex_hull_image
        return convex_hull_image(mask)
    except ImportError:
        pass
    pts = np.column_stack(np.where(mask.astype(bool)))
    if pts.shape[0] < 3:
        return mask.astype(bool)
    hull = cv2.convexHull(pts[:, ::-1].astype(np.float32))
    out = np.zeros(mask.shape, dtype=np.uint8)
    cv2.fillConvexPoly(out, hull.astype(np.int32), 1)
    return out.astype(bool)


def _disk_kernel(radius: int) -> np.ndarray:
    r = radius
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y <= r * r).astype(np.uint8)


# ---------------------------------------------------------------------------
# Helper 1: get_breast_mask
# ---------------------------------------------------------------------------

def get_breast_mask(img: np.ndarray) -> np.ndarray:
    """Otsu-fraction threshold + largest connected component + convex hull.

    Mirrors create_breast_mask in roi_metrics_dicom_3.0.py.
    Returns a boolean array of shape (H, W).
    """
    x = _normalize_for_mask(img)
    positive = x[x > 0]
    if positive.size == 0:
        return np.zeros(x.shape, dtype=bool)

    x_smooth = ndi.gaussian_filter(x, sigma=3.0)
    thr = _otsu_threshold(positive) * 0.15
    mask = x_smooth > thr

    mask = cv2.morphologyEx(
        mask.astype(np.uint8), cv2.MORPH_CLOSE, _disk_kernel(15)
    ).astype(bool)
    mask = _remove_small_objects(mask, min_size=4096)
    mask = _largest_component(mask)
    mask = _remove_small_holes(mask, area_threshold=8192)

    if np.any(mask):
        mask = _convex_hull_image(mask)
        bg_clip = cv2.morphologyEx(
            (x > 0.01).astype(np.uint8), cv2.MORPH_CLOSE, _disk_kernel(5)
        ).astype(bool)
        mask = mask & bg_clip

    return mask.astype(bool)


# ---------------------------------------------------------------------------
# Helper 2: get_sigma0
# ---------------------------------------------------------------------------

def get_sigma0(img: np.ndarray, mask: np.ndarray) -> float:
    """Std of pixel intensities in the EDT-selected homogeneous tissue patch.

    Replicates select_homogeneous_tissue_patch from roi_metrics_dicom_3.0.py.
    Falls back to std over the whole breast mask if no patch can be found.
    """
    patch_size = 96
    contour_margin_px = 80
    step = max(8, patch_size // 4)

    x = np.asarray(img, dtype=np.float32)
    h, w = x.shape

    x_g = ndi.gaussian_filter(x, sigma=1.0)
    gx = ndi.sobel(x_g, axis=1)
    gy = ndi.sobel(x_g, axis=0)
    grad = np.hypot(gx, gy)

    padded = np.pad(mask.astype(bool), pad_width=1, constant_values=False)
    dist = ndi.distance_transform_edt(padded)[1:-1, 1:-1]

    margins = sorted(
        {contour_margin_px,
         max(0, contour_margin_px // 2),
         max(0, contour_margin_px // 4),
         0},
        reverse=True,
    )

    for margin in margins:
        candidate = dist >= margin
        if candidate.sum() < patch_size * patch_size:
            continue

        best_score = np.inf
        best_patch: np.ndarray | None = None

        for y1 in range(0, h - patch_size + 1, step):
            for x1 in range(0, w - patch_size + 1, step):
                y2, x2 = y1 + patch_size, x1 + patch_size
                if candidate[y1:y2, x1:x2].mean() < 1.0:
                    continue
                score = float(grad[y1:y2, x1:x2].mean())
                if score < best_score:
                    best_score = score
                    best_patch = img[y1:y2, x1:x2]

        if best_patch is not None:
            return float(np.std(best_patch.astype(np.float32)))

    # Fallback: use std over the whole breast mask
    return float(np.std(img[mask.astype(bool)].astype(np.float32)))


# ---------------------------------------------------------------------------
# Helper 3: get_pixel_spacing
# ---------------------------------------------------------------------------

def get_pixel_spacing(dicom_metadata: dict[str, Any]) -> float:
    """Read pixel spacing (mm) from a TAR-shard JSON metadata dict.

    Tries 'PixelSpacing' then 'ImagerPixelSpacing' (as stored by pydicom
    in the JSON sidecars written by dicom_to_tar_shards.py).
    Returns 0.07 mm (70 µm, typical VinDr-Mammo Siemens) if not found.
    """
    for key in ("PixelSpacing", "ImagerPixelSpacing"):
        val = dicom_metadata.get(key)
        if val is None:
            continue
        if isinstance(val, (list, tuple)) and len(val) >= 1:
            try:
                return float(val[0])
            except (TypeError, ValueError):
                continue
        try:
            return float(val)
        except (TypeError, ValueError):
            continue
    return 0.07


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEGRADATION_TYPES = ["noise", "blur", "contrast", "jpeg2000", "resolution"]
SEVERITY_LEVELS   = [1, 2, 3, 4, 5, 6]


# ---------------------------------------------------------------------------
# Function 1: dose_noise
# ---------------------------------------------------------------------------

def dose_noise(
    img: np.ndarray,
    severity: int,
    mask: np.ndarray,
    sigma0: float,
    seed: int = 0,
) -> np.ndarray:
    """Additive Gaussian noise motivated by dose reduction (eq 4.7–4.9).

    Parameters
    ----------
    img      : float32 (H, W) in [0, 1]
    severity : 1–6
    mask     : boolean breast mask (H, W) — used by caller for sigma0 estimation
    sigma0   : std of pixel intensities in the homogeneous tissue patch
    seed     : per-image RNG seed for reproducibility
    """
    dose_factors = [0.95, 0.85, 0.75, 0.65, 0.50, 0.35]
    f_D = dose_factors[severity - 1]
    sigma_target = sigma0 / math.sqrt(f_D)                         # eq 4.7
    sigma_add = math.sqrt(max(sigma_target ** 2 - sigma0 ** 2, 0.0))  # eq 4.8

    if sigma_add <= 0.0:
        return np.asarray(img, dtype=np.float32).copy()

    rng = np.random.default_rng(seed)
    eta = rng.normal(0.0, sigma_add, size=img.shape).astype(np.float32)
    return np.clip(np.asarray(img, dtype=np.float32) + eta, 0.0, 1.0)  # eq 4.9


# ---------------------------------------------------------------------------
# Function 2: motion_blur
# ---------------------------------------------------------------------------

def motion_blur(
    img: np.ndarray,
    severity: int,
    pixel_spacing_mm: float = 0.07,
) -> np.ndarray:
    """Horizontal (θ=0°) motion blur with explicit pixel-unit kernel lengths (eq 4.11).

    Parameters
    ----------
    img              : float32 (H, W) in [0, 1]
    severity         : 1–6
    pixel_spacing_mm : kept for API compatibility only — NOT used to derive
                       the kernel length (see note below).

    Note
    ----
    Kernel length is defined directly in PREPROCESSED pixel units (the
    1024x384 canvas), not via a physical-mm length divided by effective
    pixel spacing. The mm-based mapping collapsed several severities to the
    same effective kernel once corrected for each image's effective pixel
    spacing (e.g. severities 1-3, and for many images 1-4, all rounded to a
    3 px kernel) — making those severities produce identical predictions.
    Explicit odd kernel lengths guarantee every severity is distinct.
    """
    kernel_lengths_px = [3, 5, 7, 11, 15, 21]                       # eq 4.11
    L_px = kernel_lengths_px[severity - 1]

    kernel = np.zeros((L_px, L_px), dtype=np.float32)
    kernel[L_px // 2, :] = 1.0 / L_px

    out = cv2.filter2D(
        np.asarray(img, dtype=np.float32),
        ddepth=-1,
        kernel=kernel,
        borderType=cv2.BORDER_REFLECT_101,
    )
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Function 3: contrast_reduction
# ---------------------------------------------------------------------------

def contrast_reduction(
    img: np.ndarray,
    severity: int,
    mask: np.ndarray,
) -> np.ndarray:
    """Breast-masked mean-centering contrast compression (eq 4.12–4.13).

    Parameters
    ----------
    img      : float32 (H, W) in [0, 1]
    severity : 1–6
    mask     : boolean breast mask (H, W)
    """
    alphas = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70]
    alpha = alphas[severity - 1]
    m = mask.astype(bool)
    if not np.any(m):
        return np.asarray(img, dtype=np.float32).copy()
    c_B = float(np.median(img[m]))                                  # eq 4.12
    out = np.asarray(img, dtype=np.float32).copy()
    out[m] = c_B + alpha * (img[m] - c_B)                          # eq 4.13
    return np.clip(out, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Function 4: jpeg2000
# ---------------------------------------------------------------------------

def jpeg2000(img: np.ndarray, severity: int) -> np.ndarray:
    """JPEG2000 lossy encode-decode at compression ratio CR (eq 4.14–4.15).

    Requires glymur. Falls back to JPEG via cv2 with a warning if unavailable.
    """
    compression_ratios = [10, 25, 50, 100, 250, 500]
    CR = compression_ratios[severity - 1]

    img_u16 = np.round(
        np.clip(np.asarray(img, dtype=np.float32), 0.0, 1.0) * 65535
    ).astype(np.uint16)                                             # eq 4.14

    try:
        import glymur  # type: ignore
        # mktemp returns a path without creating the file; NamedTemporaryFile
        # creates an empty file which triggers a struct.error in glymur 0.14
        # when it tries to parse the empty JP2 header before writing.
        tmp = tempfile.mktemp(suffix=".jp2")
        try:
            glymur.Jp2k(tmp, data=img_u16, cratios=[CR])
            decoded = glymur.Jp2k(tmp)[:]
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        return np.clip(decoded.astype(np.float32) / 65535.0, 0.0, 1.0)  # eq 4.15

    except Exception as exc:
        warnings.warn(
            f"jpeg2000: glymur failed ({exc}); falling back to JPEG (not JP2). "
            "Install glymur for accurate JPEG2000 compression."
        )

    # JPEG fallback — approximate quality mapped from CR
    quality_map = {10: 95, 25: 80, 50: 60, 100: 40, 250: 20, 500: 5}
    quality = quality_map.get(CR, max(5, min(95, int(100 - CR / 5))))
    img_u8 = (np.clip(np.asarray(img, dtype=np.float32), 0.0, 1.0) * 255).astype(np.uint8)
    _, buf = cv2.imencode(".jpg", img_u8, [cv2.IMWRITE_JPEG_QUALITY, quality])
    dec = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    return dec.astype(np.float32) / 255.0


# ---------------------------------------------------------------------------
# Function 5: resolution_reduction
# ---------------------------------------------------------------------------

def resolution_reduction(img: np.ndarray, severity: int) -> np.ndarray:
    """INTER_AREA downscale then INTER_CUBIC upsample (eq 4.16).

    Parameters
    ----------
    img      : float32 (H, W) in [0, 1]
    severity : 1–6
    """
    factors = [0.90, 0.80, 0.70, 0.60, 0.50, 0.40]
    r = factors[severity - 1]
    H, W = img.shape
    H_small = max(1, int(math.floor(r * H)))
    W_small = max(1, int(math.floor(r * W)))
    x = np.asarray(img, dtype=np.float32)
    small = cv2.resize(x, (W_small, H_small), interpolation=cv2.INTER_AREA)
    out   = cv2.resize(small, (W, H),          interpolation=cv2.INTER_CUBIC)  # eq 4.16
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def apply_degradation(
    img: np.ndarray,
    degradation_type: str,
    severity: int,
    mask: np.ndarray | None = None,
    sigma0: float | None = None,
    pixel_spacing_mm: float = 0.07,
    seed: int = 0,
) -> np.ndarray:
    """Dispatch to one of the five degradation functions by name.

    If mask or sigma0 are None and the degradation needs them,
    they are computed internally via get_breast_mask / get_sigma0.

    Parameters
    ----------
    img              : float32 (H, W) in [0, 1]
    degradation_type : one of DEGRADATION_TYPES
    severity         : 1–6
    mask             : pre-computed breast mask (optional, avoids recompute)
    sigma0           : pre-computed tissue noise std (optional, avoids recompute)
    pixel_spacing_mm : detector pixel pitch for motion_blur (default 0.07 mm)
    seed             : per-image RNG seed for dose_noise
    """
    if degradation_type not in DEGRADATION_TYPES:
        raise ValueError(
            f"Unknown degradation_type {degradation_type!r}. "
            f"Choose from {DEGRADATION_TYPES}."
        )

    if degradation_type == "noise":
        if mask is None:
            mask = get_breast_mask(img)
        if sigma0 is None:
            sigma0 = get_sigma0(img, mask)
        return dose_noise(img, severity, mask, sigma0, seed=seed)

    if degradation_type == "blur":
        return motion_blur(img, severity, pixel_spacing_mm=pixel_spacing_mm)

    if degradation_type == "contrast":
        if mask is None:
            mask = get_breast_mask(img)
        return contrast_reduction(img, severity, mask)

    if degradation_type == "jpeg2000":
        return jpeg2000(img, severity)

    if degradation_type == "resolution":
        return resolution_reduction(img, severity)

    raise RuntimeError(f"Unhandled degradation_type: {degradation_type!r}")


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    img_test    = np.random.default_rng(42).random((1024, 384)).astype(np.float32)
    mask_test   = img_test > 0.1
    sigma0_test = float(np.std(img_test[mask_test]))

    for fn_name, fn_kwargs in [
        ("noise",       {"mask": mask_test, "sigma0": sigma0_test}),
        ("blur",        {"pixel_spacing_mm": 0.07}),
        ("contrast",    {"mask": mask_test}),
        ("jpeg2000",    {}),
        ("resolution",  {}),
    ]:
        for s in range(1, 7):
            out = apply_degradation(img_test, fn_name, s, **fn_kwargs)
            assert out.shape == img_test.shape, f"{fn_name} s={s}: shape mismatch"
            assert out.dtype == np.float32,     f"{fn_name} s={s}: dtype mismatch"
            assert out.min() >= 0.0,            f"{fn_name} s={s}: below 0"
            assert out.max() <= 1.0,            f"{fn_name} s={s}: above 1"
            assert not np.isnan(out).any(),     f"{fn_name} s={s}: NaN"
            print(f"  {fn_name:12s} s={s}  min={out.min():.4f}  max={out.max():.4f}  "
                  f"mean={out.mean():.4f}  std={out.std():.4f}")

    print("All tests passed.")
