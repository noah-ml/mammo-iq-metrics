from __future__ import annotations

"""
VinDr-Mammo ROI and image-quality toolkit (DICOM-first)
=======================================================

This module is designed for thesis workflows on VinDr-Mammo and similar
mammography datasets. It supports:
- DICOM-first image loading with PNG/JPG fallback
- Breast mask creation on raw/full-resolution images
- Breast crop extraction
- Fixed normalization estimated ONCE from the original crop
- Fixed ROI selection on the original image only
- ROI-based metrics for degraded variants using the SAME ROIs
- QC overlay export

Degradation types (see SEVERITY_PRESETS for default severity levels)
---------------------------------------------------------------------
noise        Dose-reduction-equivalent additive Gaussian noise. Severity is
             defined by a dose reduction factor, not an arbitrary sigma.
             Requires a per-image baseline_noise_std (estimated from the
             original tissue patch).
motion_blur  Directional linear PSF (horizontal by default). Physically
             parameterised in mm when PixelSpacing is available from the DICOM
             header; falls back to pixel-based otherwise.
contrast     Linear contrast compression around the breast-region median.
             Conservative alpha values (0.95–0.70) keep degradation realistic.

Metrics computed by compute_fixed_roi_metrics
---------------------------------------------
All images:
  noise_variance  — variance of the fixed homogeneous tissue patch
  tenengrad       — mean squared Sobel gradient over breast mask (sharpness)
  ssim            — full-reference SSIM vs. original, masked to breast region

Lesion images only (when annotation bounding boxes are available):
  cnr             — |lesion_mean - bg_mean| / bg_std (background ring)
  delta_mu        — |lesion_mean - bg_mean| (absolute contrast)
  cnr_lesion_mean, cnr_bg_mean, cnr_bg_std, cnr_bg_pixels — CNR sub-components

Intensity convention
--------------------
All downstream metric and degradation functions expect normalized images in
float32 with range [0, 1]. The fixed normalization parameters must be estimated
from the original image and then reused for all degraded variants of that image.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage import filters, measure, morphology
from skimage.metrics import structural_similarity
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import warnings


ArrayLike = np.ndarray
BBox = tuple[int, int, int, int]  # x_min, y_min, x_max, y_max


@dataclass
class ImageLoadResult:
    image: ArrayLike
    source_path: str
    source_type: str
    photometric_interpretation: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class NormalizationParams:
    low: float
    high: float
    lower_percentile: float
    upper_percentile: float
    based_on_mask: bool


@dataclass
class BreastCropResult:
    image: ArrayLike
    mask: ArrayLike
    bbox: tuple[int, int, int, int]  # y_min, y_max, x_min, x_max


@dataclass
class TissuePatchResult:
    patch: ArrayLike
    bbox: BBox
    score: float
    contour_margin_used_px: int


@dataclass
class CNRResult:
    lesion_bbox: BBox
    bg_bbox_outer: BBox
    cnr: float
    lesion_mean: float
    bg_mean: float
    bg_std: float
    num_bg_pixels: int


@dataclass
class DegradationSpec:
    degradation_type: str
    severity: int
    params: dict[str, Any]


# -----------------------------------------------------------------------------
# I/O
# -----------------------------------------------------------------------------

def _is_dicom_path(path: str | Path) -> bool:
    suffix = Path(path).suffix.lower()
    return suffix in {".dcm", ".dicom"}


def read_mammogram(path: str | Path, dicom_force_invert: bool | None = None) -> ImageLoadResult:
    """Read a mammogram from DICOM, PNG, JPG, or JPEG.

    DICOM handling notes:
    - Applies RescaleSlope / RescaleIntercept if present.
    - Automatically inverts MONOCHROME1 unless ``dicom_force_invert`` overrides it.
    - Returns a raw float32 pixel array, not normalized.
    """
    path = Path(path)
    if _is_dicom_path(path):
        try:
            import pydicom  # type: ignore
        except Exception as e:  # pragma: no cover - depends on environment
            raise ImportError(
                "DICOM loading requires 'pydicom'. Install it with: pip install pydicom"
            ) from e

        ds = pydicom.dcmread(str(path))
        arr = ds.pixel_array
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim > 2:
            arr = np.squeeze(arr)
            if arr.ndim != 2:
                raise ValueError(f"Unsupported DICOM pixel array shape: {arr.shape}")

        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        arr = arr * slope + intercept

        photometric = str(getattr(ds, "PhotometricInterpretation", "") or "")
        should_invert = False
        if dicom_force_invert is None:
            should_invert = photometric.upper() == "MONOCHROME1"
        else:
            should_invert = bool(dicom_force_invert)

        if should_invert:
            arr = arr.max() - arr

        # PixelSpacing is stored as [row_spacing, col_spacing] in mm/pixel.
        # We take the column spacing (x-axis) as the reference for motion blur
        # length conversion, since horizontal motion is the dominant artifact.
        pixel_spacing_mm: float | None = None
        raw_ps = getattr(ds, "PixelSpacing", None) or getattr(ds, "ImagerPixelSpacing", None)
        if raw_ps is not None:
            try:
                pixel_spacing_mm = float(raw_ps[1])  # col spacing (x-axis)
            except (IndexError, TypeError, ValueError):
                pixel_spacing_mm = None

        metadata = {
            "Rows": int(getattr(ds, "Rows", arr.shape[0])),
            "Columns": int(getattr(ds, "Columns", arr.shape[1])),
            "BitsStored": getattr(ds, "BitsStored", None),
            "PhotometricInterpretation": photometric,
            "RescaleSlope": slope,
            "RescaleIntercept": intercept,
            "SOPInstanceUID": str(getattr(ds, "SOPInstanceUID", "") or ""),
            "PixelSpacing_mm": pixel_spacing_mm,
        }
        return ImageLoadResult(
            image=arr.astype(np.float32, copy=False),
            source_path=str(path),
            source_type="dicom",
            photometric_interpretation=photometric,
            metadata=metadata,
        )

    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=np.float32)
    return ImageLoadResult(
        image=arr,
        source_path=str(path),
        source_type="image",
        photometric_interpretation=None,
        metadata={"mode": "L"},
    )


# -----------------------------------------------------------------------------
# Normalization helpers
# -----------------------------------------------------------------------------

def normalize_for_mask(img: ArrayLike) -> ArrayLike:
    """Simple min-max normalization to [0, 1] for mask creation only."""
    x = np.asarray(img, dtype=np.float32)
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x, dtype=np.float32)
    vals = x[finite]
    mn, mx = float(vals.min()), float(vals.max())
    if mx <= mn:
        return np.zeros_like(x, dtype=np.float32)
    out = (x - mn) / (mx - mn)
    out[~finite] = 0.0
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def estimate_fixed_normalization(
    img: ArrayLike,
    mask: ArrayLike | None = None,
    lower_percentile: float = 0.5,
    upper_percentile: float = 99.5,
) -> NormalizationParams:
    """Estimate one robust intensity window from the ORIGINAL image.

    The returned parameters should then be reused for all degraded variants of
    the same image.
    """
    x = np.asarray(img, dtype=np.float32)
    if mask is not None and np.any(mask):
        vals = x[mask.astype(bool)]
        based_on_mask = True
    else:
        vals = x.ravel()
        based_on_mask = False

    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return NormalizationParams(0.0, 1.0, lower_percentile, upper_percentile, based_on_mask)

    low = float(np.percentile(vals, lower_percentile))
    high = float(np.percentile(vals, upper_percentile))
    if high <= low:
        low = float(vals.min())
        high = float(vals.max())
    if high <= low:
        high = low + 1.0

    return NormalizationParams(low, high, lower_percentile, upper_percentile, based_on_mask)


def apply_fixed_normalization(img: ArrayLike, params: NormalizationParams) -> ArrayLike:
    x = np.asarray(img, dtype=np.float32)
    out = (x - params.low) / (params.high - params.low)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# -----------------------------------------------------------------------------
# Breast mask / crop / ROI helpers
# -----------------------------------------------------------------------------

def _largest_component(mask: ArrayLike) -> ArrayLike:
    labels = measure.label(mask.astype(bool), connectivity=2)
    props = measure.regionprops(labels)
    if not props:
        return np.zeros_like(mask, dtype=bool)
    largest = max(props, key=lambda p: p.area)
    return labels == largest.label


def create_breast_mask(
    img: ArrayLike,
    threshold: str = "otsu_fraction",
    closing_radius: int = 15,
    opening_radius: int = 0,
    min_hole_size: int = 8192,
    min_object_size: int = 4096,
    otsu_fraction: float = 0.15,
    use_convex_hull: bool = True,
    smooth_sigma: float = 3.0,
) -> ArrayLike:
    """Create a breast mask from the raw image.

    This is intentionally computed on the original image, not on a degraded one.
    """
    x = normalize_for_mask(img)
    positive = x[x > 0]
    if positive.size == 0:
        return np.zeros_like(x, dtype=bool)

    if smooth_sigma > 0:
        x_smooth = filters.gaussian(x, sigma=smooth_sigma, preserve_range=True)
    else:
        x_smooth = x

    if threshold == "otsu_fraction":
        thr = float(filters.threshold_otsu(positive)) * otsu_fraction
    elif threshold == "otsu_positive":
        thr = float(filters.threshold_otsu(positive))
    elif threshold == "yen_positive":
        thr = float(filters.threshold_yen(positive))
    elif threshold == "percentile":
        thr = float(np.percentile(positive, 2.0))
    elif threshold == "nonzero":
        thr = 0.0
    else:
        raise ValueError(f"Unknown threshold strategy: {threshold}")

    mask = x_smooth > thr
    if closing_radius > 0:
        mask = morphology.closing(mask, morphology.disk(closing_radius))
    if opening_radius > 0:
        mask = morphology.opening(mask, morphology.disk(opening_radius))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        mask = morphology.remove_small_objects(mask, min_size=min_object_size)
        mask = _largest_component(mask)
        mask = morphology.remove_small_holes(mask, area_threshold=min_hole_size)

    if use_convex_hull and np.any(mask):
        mask = morphology.convex_hull_image(mask)
        bg_clip = x > 0.01
        bg_clip = morphology.closing(bg_clip, morphology.disk(5))
        mask = mask & bg_clip

    return mask.astype(bool)


def crop_to_breast(img: ArrayLike, mask: ArrayLike, buffer_px: int = 20) -> BreastCropResult:
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        raise ValueError("Mask is empty; cannot crop.")

    y_min = max(0, int(ys.min()) - buffer_px)
    y_max = min(img.shape[0], int(ys.max()) + 1 + buffer_px)
    x_min = max(0, int(xs.min()) - buffer_px)
    x_max = min(img.shape[1], int(xs.max()) + 1 + buffer_px)

    return BreastCropResult(
        image=np.asarray(img[y_min:y_max, x_min:x_max]).copy(),
        mask=np.asarray(mask[y_min:y_max, x_min:x_max]).copy(),
        bbox=(y_min, y_max, x_min, x_max),
    )


def _clip_bbox(bbox: BBox, h: int, w: int) -> BBox:
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(w, int(x1)))
    x2 = max(0, min(w, int(x2)))
    y1 = max(0, min(h, int(y1)))
    y2 = max(0, min(h, int(y2)))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid bbox after clipping: {bbox}")
    return x1, y1, x2, y2


def shift_bboxes_to_crop(bboxes: Sequence[BBox], crop_bbox: tuple[int, int, int, int]) -> list[BBox]:
    y_min, y_max, x_min, x_max = crop_bbox
    h = y_max - y_min
    w = x_max - x_min
    shifted: list[BBox] = []
    for x1, y1, x2, y2 in bboxes:
        try:
            sb = _clip_bbox((x1 - x_min, y1 - y_min, x2 - x_min, y2 - y_min), h, w)
            shifted.append(sb)
        except ValueError:
            # silently ignore boxes fully outside the crop
            continue
    return shifted


def bbox_mask(shape: tuple[int, int], bbox: BBox) -> ArrayLike:
    h, w = shape
    x1, y1, x2, y2 = _clip_bbox(bbox, h, w)
    m = np.zeros((h, w), dtype=bool)
    m[y1:y2, x1:x2] = True
    return m


def dilate_bboxes_mask(shape: tuple[int, int], bboxes: Sequence[BBox], dilation_px: int = 16) -> ArrayLike:
    m = np.zeros(shape, dtype=bool)
    for bbox in bboxes:
        m |= bbox_mask(shape, bbox)
    if dilation_px > 0 and np.any(m):
        m = morphology.dilation(m, morphology.disk(dilation_px))
    return m


def select_primary_lesion_bbox(lesion_bboxes: Sequence[BBox]) -> BBox | None:
    if not lesion_bboxes:
        return None
    return max(lesion_bboxes, key=lambda b: max(0, b[2] - b[0]) * max(0, b[3] - b[1]))


def select_homogeneous_tissue_patch(
    img: ArrayLike,
    breast_mask: ArrayLike,
    lesion_bboxes: Sequence[BBox] | None = None,
    patch_size: int = 96,
    contour_margin_px: int = 80,
    lesion_margin_px: int = 16,
    gradient_sigma: float = 1.0,
    min_mask_coverage: float = 1.0,
) -> TissuePatchResult:
    """Select a low-gradient, lesion-free tissue patch on the ORIGINAL image.

    ``img`` is expected to be the fixed-normalized original crop in [0, 1].
    """
    h, w = img.shape
    if patch_size >= min(h, w):
        raise ValueError("Patch size is too large for the cropped image.")

    x = np.asarray(img, dtype=np.float32)
    if gradient_sigma > 0:
        x = filters.gaussian(x, sigma=gradient_sigma, preserve_range=True)
    gx = ndi.sobel(x, axis=1)
    gy = ndi.sobel(x, axis=0)
    grad = np.hypot(gx, gy)

    padded = np.pad(breast_mask.astype(bool), pad_width=1, constant_values=False)
    dist_padded = ndi.distance_transform_edt(padded)
    dist = dist_padded[1:-1, 1:-1]

    lesion_exclusion = None
    if lesion_bboxes:
        lesion_exclusion = dilate_bboxes_mask(img.shape, lesion_bboxes, dilation_px=lesion_margin_px)

    margins_to_try = sorted({
        int(contour_margin_px),
        max(0, int(round(contour_margin_px * 0.5))),
        max(0, int(round(contour_margin_px * 0.25))),
        0,
    }, reverse=True)

    for margin in margins_to_try:
        candidate_mask = dist >= margin
        if lesion_exclusion is not None:
            candidate_mask &= ~lesion_exclusion
        if candidate_mask.sum() < patch_size * patch_size:
            continue

        best_score = np.inf
        best_bbox: BBox | None = None
        step = max(8, patch_size // 4)

        for y1 in range(0, h - patch_size + 1, step):
            for x1 in range(0, w - patch_size + 1, step):
                y2, x2 = y1 + patch_size, x1 + patch_size
                patch_mask = candidate_mask[y1:y2, x1:x2]
                if patch_mask.mean() < min_mask_coverage:
                    continue
                score = float(grad[y1:y2, x1:x2].mean())
                if score < best_score:
                    best_score = score
                    best_bbox = (x1, y1, x2, y2)

        if best_bbox is not None:
            x1, y1, x2, y2 = best_bbox
            return TissuePatchResult(
                patch=np.asarray(img[y1:y2, x1:x2]).copy(),
                bbox=best_bbox,
                score=best_score,
                contour_margin_used_px=margin,
            )

    raise RuntimeError(
        "Could not find a homogeneous tissue patch even with reduced margins. "
        "Try a smaller patch_size or inspect the breast mask."
    )


def lesion_background_ring(
    lesion_bbox: BBox,
    image_shape: tuple[int, int],
    breast_mask: ArrayLike,
    other_lesions: Sequence[BBox] | None = None,
    inner_scale: float = 1.5,
    outer_scale: float = 2.5,
) -> tuple[ArrayLike, ArrayLike, BBox]:
    h, w = image_shape
    x1, y1, x2, y2 = lesion_bbox
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    bw = x2 - x1
    bh = y2 - y1

    def scaled_box(scale: float) -> BBox:
        nw = max(1, int(round(bw * scale)))
        nh = max(1, int(round(bh * scale)))
        sx1 = int(round(cx - nw / 2))
        sy1 = int(round(cy - nh / 2))
        sx2 = int(round(cx + nw / 2))
        sy2 = int(round(cy + nh / 2))
        return _clip_bbox((sx1, sy1, sx2, sy2), h, w)

    inner = scaled_box(inner_scale)
    outer = scaled_box(outer_scale)

    lesion_m = bbox_mask((h, w), lesion_bbox)
    inner_m = bbox_mask((h, w), inner)
    outer_m = bbox_mask((h, w), outer)
    ring = outer_m & (~inner_m) & breast_mask.astype(bool)

    if other_lesions:
        other_mask = np.zeros((h, w), dtype=bool)
        for b in other_lesions:
            if tuple(b) != tuple(lesion_bbox):
                other_mask |= bbox_mask((h, w), b)
        ring &= ~other_mask

    return lesion_m, ring, outer


# -----------------------------------------------------------------------------
# Metrics on fixed ROIs
# -----------------------------------------------------------------------------

def patch_from_bbox(img: ArrayLike, bbox: BBox) -> ArrayLike:
    x1, y1, x2, y2 = bbox
    return np.asarray(img[y1:y2, x1:x2], dtype=np.float32)


def compute_noise_variance_from_patch(patch: ArrayLike) -> float:
    return float(np.var(np.asarray(patch, dtype=np.float32), ddof=0))


def compute_cnr(img: ArrayLike, lesion_bbox: BBox, ring_mask: ArrayLike) -> CNRResult:
    lesion_m = bbox_mask(img.shape, lesion_bbox)
    lesion_vals = np.asarray(img[lesion_m], dtype=np.float32)
    bg_vals = np.asarray(img[ring_mask.astype(bool)], dtype=np.float32)
    if lesion_vals.size == 0 or bg_vals.size == 0:
        raise ValueError("Lesion ROI or background ring is empty; cannot compute CNR.")
    lesion_mean = float(np.mean(lesion_vals))
    bg_mean = float(np.mean(bg_vals))
    bg_std = float(np.std(bg_vals, ddof=0))
    cnr = float("inf") if bg_std == 0 else abs(lesion_mean - bg_mean) / bg_std

    ys, xs = np.where(ring_mask)
    outer = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    return CNRResult(
        lesion_bbox=tuple(int(v) for v in lesion_bbox),
        bg_bbox_outer=outer,
        cnr=float(cnr),
        lesion_mean=lesion_mean,
        bg_mean=bg_mean,
        bg_std=bg_std,
        num_bg_pixels=int(bg_vals.size),
    )


def compute_tenengrad(img: ArrayLike, mask: ArrayLike | None = None) -> float:
    x = np.asarray(img, dtype=np.float32)
    gx = ndi.sobel(x, axis=1)
    gy = ndi.sobel(x, axis=0)
    g2 = gx * gx + gy * gy
    vals = g2[mask.astype(bool)] if mask is not None else g2.ravel()
    return float(np.mean(vals))


def _crop_to_mask_extent(img: ArrayLike, mask: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
    ys, xs = np.where(mask.astype(bool))
    if ys.size == 0 or xs.size == 0:
        raise ValueError("Mask is empty.")
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    img_c = np.asarray(img[y1:y2, x1:x2], dtype=np.float32)
    mask_c = np.asarray(mask[y1:y2, x1:x2], dtype=bool)
    return img_c, mask_c


def compute_ssim_fullref(img_ref: ArrayLike, img_deg: ArrayLike, mask: ArrayLike | None = None) -> float:
    ref = np.asarray(img_ref, dtype=np.float32)
    deg = np.asarray(img_deg, dtype=np.float32)
    if mask is None:
        return float(structural_similarity(ref, deg, data_range=1.0))

    ref_c, mask_c = _crop_to_mask_extent(ref, mask)
    deg_c, _ = _crop_to_mask_extent(deg, mask)
    ref_c = ref_c * mask_c
    deg_c = deg_c * mask_c
    return float(structural_similarity(ref_c, deg_c, data_range=1.0))


def compute_fixed_roi_metrics(
    original_img: ArrayLike,
    degraded_img: ArrayLike,
    breast_mask: ArrayLike,
    tissue_patch_bbox: BBox,
    lesion_bbox: BBox | None = None,
    ring_mask: ArrayLike | None = None,
) -> dict[str, float]:
    """Compute thesis metrics using ROIs fixed on the original image.

    Always computed (all images):
        noise_variance, tenengrad, ssim

    Computed only when lesion_bbox and a non-empty ring_mask are provided:
        cnr, cnr_lesion_mean, cnr_bg_mean, cnr_bg_std, cnr_bg_pixels, delta_mu
        (set to NaN otherwise)
    """
    metrics: dict[str, float] = {}

    patch = patch_from_bbox(degraded_img, tissue_patch_bbox)
    metrics["noise_variance"] = compute_noise_variance_from_patch(patch)
    metrics["tenengrad"] = compute_tenengrad(degraded_img, mask=breast_mask)

    if lesion_bbox is not None and ring_mask is not None and np.any(ring_mask):
        cnr_res = compute_cnr(degraded_img, lesion_bbox, ring_mask)
        metrics["cnr"] = cnr_res.cnr
        metrics["cnr_lesion_mean"] = cnr_res.lesion_mean
        metrics["cnr_bg_mean"] = cnr_res.bg_mean
        metrics["cnr_bg_std"] = cnr_res.bg_std
        metrics["cnr_bg_pixels"] = float(cnr_res.num_bg_pixels)
        metrics["delta_mu"] = abs(cnr_res.lesion_mean - cnr_res.bg_mean)
    else:
        metrics["cnr"] = np.nan
        metrics["cnr_lesion_mean"] = np.nan
        metrics["cnr_bg_mean"] = np.nan
        metrics["cnr_bg_std"] = np.nan
        metrics["cnr_bg_pixels"] = np.nan
        metrics["delta_mu"] = np.nan

    metrics["ssim"] = compute_ssim_fullref(original_img, degraded_img, mask=breast_mask)
    return metrics


# =============================================================================
# Severity presets
# =============================================================================

SEVERITY_PRESETS: dict[str, list] = {
    # Physical motion blur lengths in mm (converted to px via PixelSpacing).
    # Fallback pixel values pre-computed at 0.07 mm/px (VinDr-Mammo typical).
    "motion_blur_mm":          [0.10, 0.25, 0.50, 0.75, 1.00, 1.50],
    "motion_blur_px_fallback": [   3,    5,    7,   11,   15,   21],
    # Dose reduction factors relative to the original acquisition (1.0 = full dose).
    # Quantum noise variance scales as 1/dose, so lower values → more noise.
    "dose_factors":            [0.95, 0.85, 0.75, 0.65, 0.50, 0.35],
    # Linear contrast compression factors (1.0 = no change, 0.0 = flat image).
    # Conservative range: mildest degradation first.
    "contrast_alpha":          [0.95, 0.90, 0.85, 0.80, 0.75, 0.70],
}


# =============================================================================
# Degradation functions
# =============================================================================

def clip01(x: ArrayLike) -> ArrayLike:
    """Clip a float32 array to [0, 1]."""
    return np.clip(np.asarray(x, dtype=np.float32), 0.0, 1.0)


# --- Motion blur -------------------------------------------------------------

def build_motion_blur_kernel(length_px: int, angle: float = 0.0) -> np.ndarray:
    """Build a normalised linear (line-spread) motion blur kernel.

    Models a uniform translational exposure blur as a straight line of equal
    weights, i.e. a rectangular line-spread PSF. The kernel sums to 1.

    Parameters
    ----------
    length_px : int
        Kernel length in pixels. Forced to the nearest odd integer >= 3 so
        the kernel has an exact centre pixel.
    angle : float
        Blur direction in degrees. 0° = horizontal (default).
        For sensitivity analysis use {-10, 0, +10} degrees.

    Returns
    -------
    kernel : np.ndarray, shape (length_px, length_px), float32
    """
    import cv2  # type: ignore

    if length_px < 1:
        raise ValueError(f"length_px must be >= 1, got {length_px}")

    length_px = max(3, int(length_px))
    if length_px % 2 == 0:
        length_px += 1  # ensure odd — required for a centred PSF

    # Horizontal unit-energy line at the kernel centre row.
    kernel = np.zeros((length_px, length_px), dtype=np.float32)
    kernel[length_px // 2, :] = 1.0 / length_px

    if angle != 0.0:
        center = (length_px // 2, length_px // 2)
        M = cv2.getRotationMatrix2D(center, float(angle), 1.0)
        kernel = cv2.warpAffine(kernel, M, (length_px, length_px))
        # Re-normalise: bilinear interpolation during rotation shifts the sum.
        s = float(kernel.sum())
        if s > 0:
            kernel /= s

    return kernel


def apply_motion_blur(
    image: ArrayLike,
    *,
    length_mm: float | None = None,
    length_px: int | None = None,
    angle: float = 0.0,
    pixel_spacing_mm: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply directional motion blur to a normalised mammogram.

    Models patient motion during X-ray exposure as a uniform lateral
    displacement (line-spread PSF). Horizontal direction (angle=0) is the
    default because breathing-induced motion is the dominant artifact in
    mammography with the breast under compression.

    Physical parameterisation (preferred)
    --------------------------------------
    Provide ``length_mm`` and ``pixel_spacing_mm`` (DICOM PixelSpacing[1],
    the column spacing in mm/pixel).  The conversion is:

        length_px = length_mm / pixel_spacing_mm

    This makes severity levels physically interpretable and comparable across
    scanners with different native resolutions.

    Image-space fallback
    --------------------
    If ``pixel_spacing_mm`` is unavailable, supply ``length_px`` directly.
    Results are then image-space approximations only — not physically
    calibrated — and comparisons across resolutions are not valid.

    Parameters
    ----------
    image : array-like, float32, shape (H, W)
        Normalised grayscale mammogram in [0, 1].
    length_mm : float, optional
        Physical blur length in millimetres.
    length_px : int, optional
        Fallback blur length in pixels (used only when pixel_spacing_mm
        is None or when length_mm is None).
    angle : float
        Blur direction in degrees. 0 = horizontal (default).
    pixel_spacing_mm : float, optional
        DICOM column pixel spacing in mm/pixel (PixelSpacing[1]).

    Returns
    -------
    blurred : np.ndarray, float32
        Blurred image, clipped to [0, 1].
    kernel : np.ndarray, float32
        The kernel used — save for QC / methods documentation.

    Raises
    ------
    ValueError
        If neither a valid (length_mm + pixel_spacing_mm) pair nor a
        length_px value is provided.
    """
    import cv2  # type: ignore

    # --- Resolve kernel length in pixels ------------------------------------
    if length_mm is not None and pixel_spacing_mm is not None:
        # Physically calibrated path: mm → pixels via DICOM pixel spacing.
        if length_mm <= 0:
            raise ValueError(f"length_mm must be > 0, got {length_mm}")
        if pixel_spacing_mm <= 0:
            raise ValueError(f"pixel_spacing_mm must be > 0, got {pixel_spacing_mm}")
        resolved_px = max(3, int(round(length_mm / pixel_spacing_mm)))
    elif length_px is not None:
        # Image-space fallback — not physically calibrated.
        resolved_px = max(3, int(length_px))
    else:
        raise ValueError(
            "Provide (length_mm + pixel_spacing_mm) for a physically calibrated "
            "blur, or length_px as an image-space approximation."
        )

    kernel = build_motion_blur_kernel(resolved_px, angle=angle)

    # Convolve with BORDER_REFLECT to avoid dark-border artefacts at edges.
    x = np.asarray(image, dtype=np.float32)
    blurred = cv2.filter2D(x, -1, kernel, borderType=cv2.BORDER_REFLECT)
    return clip01(blurred), kernel


# --- Dose-based noise --------------------------------------------------------

def apply_dose_based_noise(
    image: ArrayLike,
    dose_factor: float,
    baseline_noise_std: float,
    *,
    seed: int | None = None,
) -> np.ndarray:
    """Apply dose-reduction-equivalent noise increase.

    In mammography, quantum noise variance scales approximately inversely with
    dose (Poisson statistics):

        var(noise) ∝ 1 / dose

    Given a baseline noise level from the original image, the additional noise
    required to simulate a reduced-dose acquisition is:

        target_std      = baseline_noise_std / sqrt(dose_factor)
        additional_std  = sqrt(max(target_std² − baseline_noise_std², 0))

    The result is additive Gaussian noise, which is a practical approximation
    of the Poisson quantum noise model in the normalised [0, 1] image space.
    Severity is defined by ``dose_factor``, not by an arbitrary sigma, making
    levels physically interpretable.

    Parameters
    ----------
    image : array-like, float32, shape (H, W)
        Normalised mammogram in [0, 1].
    dose_factor : float
        Relative dose in (0, 1]. 1.0 = original dose (no change);
        0.5 = half dose (substantially more noise).
    baseline_noise_std : float
        Standard deviation of noise in the original image, estimated in the
        same normalised [0, 1] space (e.g. std of a homogeneous tissue patch).
    seed : int, optional
        Random seed for deterministic output.

    Returns
    -------
    noisy : np.ndarray, float32
        Noise-augmented image clipped to [0, 1].

    Example
    -------
    >>> patch_std = float(np.std(original_patch))
    >>> for dose in [0.95, 0.85, 0.75, 0.65, 0.50, 0.35]:
    ...     noisy = apply_dose_based_noise(image, dose, patch_std, seed=42)
    """
    if not (0 < dose_factor <= 1.0):
        raise ValueError(f"dose_factor must be in (0, 1], got {dose_factor}")
    if baseline_noise_std < 0:
        raise ValueError(f"baseline_noise_std must be >= 0, got {baseline_noise_std}")

    # Target noise std at the reduced dose level.
    target_std = baseline_noise_std / np.sqrt(dose_factor)
    # Only the *additional* noise on top of the existing baseline is added.
    additional_var = target_std ** 2 - baseline_noise_std ** 2
    additional_std = float(np.sqrt(max(additional_var, 0.0)))

    x = np.asarray(image, dtype=np.float32)

    if additional_std == 0.0:
        # dose_factor == 1.0 or baseline_noise_std == 0 → no change.
        return x.copy()

    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(x.shape).astype(np.float32) * additional_std
    return clip01(x + noise)


# --- Contrast reduction ------------------------------------------------------

def apply_contrast_reduction(
    image: ArrayLike,
    alpha: float,
    *,
    mask: ArrayLike | None = None,
    center_mode: str = "masked_median",
) -> np.ndarray:
    """Apply linear contrast compression around a reference breast intensity.

    Formula:

        I_degraded = center + alpha × (I − center)

    where ``center`` is estimated from within the breast region so the
    compression is anchored to physiologically meaningful tissue intensity
    rather than the image mean (which is dominated by the black background).

    Parameters
    ----------
    image : array-like, float32, shape (H, W)
        Normalised mammogram in [0, 1].
    alpha : float
        Contrast scaling factor in (0, 1]. 1.0 = no change; values closer to
        0.0 progressively flatten contrast towards the centre intensity.
    mask : array-like of bool, optional
        Breast region mask. Used to compute the scaling centre from tissue
        pixels only.  Strongly recommended — without a mask the large black
        background biases the centre estimate towards zero.
    center_mode : str
        How to estimate the reference intensity:
        ``"masked_median"``  — median of masked pixels (default, robust).
        ``"masked_mean"``    — mean of masked pixels.
        ``"median"``         — global image median.
        ``"mean"``           — global image mean.
        ``"midpoint"``       — fixed 0.5 (image midpoint).

    Returns
    -------
    degraded : np.ndarray, float32
        Contrast-reduced image clipped to [0, 1].

    Example
    -------
    >>> for alpha in [0.95, 0.90, 0.85, 0.80, 0.75, 0.70]:
    ...     degraded = apply_contrast_reduction(image, alpha, mask=breast_mask)
    """
    if not (0 < alpha <= 1.0):
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")

    x = np.asarray(image, dtype=np.float32)

    # --- Compute scaling centre from the requested region / statistic -------
    valid_modes = {"masked_median", "masked_mean", "median", "mean", "midpoint"}
    if center_mode not in valid_modes:
        raise ValueError(f"center_mode must be one of {valid_modes}, got {center_mode!r}")

    if center_mode == "midpoint":
        center = 0.5
    elif center_mode in ("masked_median", "masked_mean") and mask is not None:
        pixels = x[np.asarray(mask, dtype=bool)]
        if pixels.size == 0:
            pixels = x.ravel()  # fallback: mask was empty
        center = float(np.median(pixels) if center_mode == "masked_median"
                       else np.mean(pixels))
    else:
        # "median", "mean", or masked mode without a mask provided.
        center = float(np.median(x) if "median" in center_mode else np.mean(x))

    # Apply: I_degraded = center + alpha * (I - center)
    out = np.asarray(center + alpha * (x - center), dtype=np.float32)
    return clip01(out)


# =============================================================================
# Pipeline dispatcher
# =============================================================================

def apply_degradation(
    image: ArrayLike,
    degradation_type: str,
    severity: int,
    *,
    pixel_spacing_mm: float | None = None,
    baseline_noise_std: float | None = None,
    seed: int | None = None,
    mask: ArrayLike | None = None,
    center_mode: str = "masked_median",
    angle: float = 0.0,
) -> np.ndarray:
    """Apply a single degradation at a given severity level.

    Severity indices (1–6) map to ``SEVERITY_PRESETS``:

        severity  motion_blur_mm  dose_factor  contrast_alpha
        --------  --------------  -----------  --------------
           1           0.10          0.95          0.95
           2           0.25          0.85          0.90
           3           0.50          0.75          0.85
           4           0.75          0.65          0.80
           5           1.00          0.50          0.75
           6           1.50          0.35          0.70

    Parameters
    ----------
    image : array-like, float32, shape (H, W)
        Normalised mammogram in [0, 1].
    degradation_type : str
        One of ``"none"``, ``"noise"``, ``"motion_blur"``, ``"contrast"``.
    severity : int
        Severity index 1–6 (ignored when degradation_type is "none").
    pixel_spacing_mm : float, optional
        DICOM column pixel spacing (mm/pixel) for physically calibrated blur.
        If None, the pre-computed fallback pixel lengths are used.
    baseline_noise_std : float, optional
        Noise std of the original tissue patch in [0, 1] space.
        Required for ``degradation_type="noise"``.
    seed : int, optional
        Random seed for noise reproducibility.
    mask : array-like of bool, optional
        Breast region mask — used as scaling centre for contrast reduction.
    center_mode : str
        Contrast centre estimation mode (see ``apply_contrast_reduction``).
    angle : float
        Motion blur direction in degrees. 0 = horizontal (default).

    Returns
    -------
    degraded : np.ndarray, float32

    Raises
    ------
    ValueError
        On unknown degradation_type, out-of-range severity, or missing
        required kwargs.
    """
    if degradation_type == "none":
        return np.asarray(image, dtype=np.float32).copy()

    if not (1 <= severity <= 6):
        raise ValueError(f"severity must be between 1 and 6, got {severity}")
    idx = severity - 1  # 0-based index into SEVERITY_PRESETS

    if degradation_type == "noise":
        if baseline_noise_std is None:
            raise ValueError(
                "baseline_noise_std is required for noise degradation. "
                "Estimate it as np.std(tissue_patch) on the original image."
            )
        return apply_dose_based_noise(
            image,
            dose_factor=SEVERITY_PRESETS["dose_factors"][idx],
            baseline_noise_std=baseline_noise_std,
            seed=seed,
        )

    if degradation_type == "motion_blur":
        blurred, _ = apply_motion_blur(
            image,
            length_mm=SEVERITY_PRESETS["motion_blur_mm"][idx],
            length_px=SEVERITY_PRESETS["motion_blur_px_fallback"][idx],
            angle=angle,
            pixel_spacing_mm=pixel_spacing_mm,
        )
        return blurred

    if degradation_type == "contrast":
        return apply_contrast_reduction(
            image,
            alpha=SEVERITY_PRESETS["contrast_alpha"][idx],
            mask=mask,
            center_mode=center_mode,
        )

    raise ValueError(
        f"Unknown degradation_type: {degradation_type!r}. "
        f"Valid types: 'none', 'noise', 'motion_blur', 'contrast'."
    )


def build_default_degradation_plan() -> list[DegradationSpec]:
    """Build the default degradation plan: 19 variants per image.

    1 baseline (none) + 6 noise + 6 motion_blur + 6 contrast.

    DegradationSpec.params stores the resolved parameter values so they are
    logged in the output CSV via ``row.update(spec.params)``.
    """
    plan = [DegradationSpec("none", 0, {})]

    for sev in range(1, 7):
        plan.append(DegradationSpec("noise", sev, {
            "dose_factor": SEVERITY_PRESETS["dose_factors"][sev - 1],
        }))

    for sev in range(1, 7):
        plan.append(DegradationSpec("motion_blur", sev, {
            "length_mm": SEVERITY_PRESETS["motion_blur_mm"][sev - 1],
            "angle": 0.0,
        }))

    for sev in range(1, 7):
        plan.append(DegradationSpec("contrast", sev, {
            "alpha": SEVERITY_PRESETS["contrast_alpha"][sev - 1],
        }))

    return plan


# -----------------------------------------------------------------------------
# QC helpers
# -----------------------------------------------------------------------------

def to_uint8_for_qc(img: ArrayLike) -> ArrayLike:
    x = np.asarray(img, dtype=np.float32)
    if x.size == 0:
        return np.zeros_like(x, dtype=np.uint8)
    if x.min() >= 0.0 and x.max() <= 1.0:
        return np.clip(x * 255.0, 0, 255).astype(np.uint8)
    x_n = normalize_for_mask(x)
    return np.clip(x_n * 255.0, 0, 255).astype(np.uint8)


def save_qc_overlay(
    image: ArrayLike,
    breast_mask: ArrayLike,
    out_path: str | Path,
    tissue_patch_bbox: BBox | None = None,
    lesion_bbox: BBox | None = None,
    ring_mask: ArrayLike | None = None,
    title: str | None = None,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(to_uint8_for_qc(image), cmap="gray")
    ax.contour(breast_mask.astype(np.uint8), levels=[0.5], colors=["lime"], linewidths=1.0)

    if tissue_patch_bbox is not None:
        x1, y1, x2, y2 = tissue_patch_bbox
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="yellow", linewidth=1.5))

    if lesion_bbox is not None:
        x1, y1, x2, y2 = lesion_bbox
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="red", linewidth=1.5))

    if ring_mask is not None and np.any(ring_mask):
        ax.contour(ring_mask.astype(np.uint8), levels=[0.5], colors=["cyan"], linewidths=1.0)

    if title:
        ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_image_png(image: ArrayLike, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(to_uint8_for_qc(image)).save(out_path)


if __name__ == "__main__":  # pragma: no cover
    print(
        "Import this module into your experiment pipeline.\n"
        "Recommended workflow:\n"
        "1) read_mammogram\n"
        "2) create_breast_mask on original\n"
        "3) crop_to_breast\n"
        "4) estimate_fixed_normalization on original crop\n"
        "5) select fixed ROIs on normalized original crop\n"
        "6) apply degradations on normalized original crop\n"
        "7) compute metrics with compute_fixed_roi_metrics\n"
    )
