#!/usr/bin/env python
"""
compute_iqms_full_dataset.py
============================

Full-dataset IQM computation pipeline for VinDr-Mammo.

Computes image quality metrics for every DICOM in the dataset, for the
baseline and all 30 degraded variants (6 noise + 6 motion_blur + 6
contrast + 6 jpeg2000 + 6 resolution severities), using the fixed-ROI
strategy from roi_metrics.py (v3.1).

Design goals
------------
- Restartable: completed image IDs are written to completed.txt after each
  successful image; the script skips them on resume.
- Incremental writes: rows are flushed to the output CSV after every image;
  a crash or OOM loses at most one image worth of rows.
- Deterministic: each image has a stable random seed derived from its sorted
  position in the full file list, so resuming produces identical results.
- Cluster-friendly: --n-shards / --shard-index split work across SLURM
  job-array tasks; use --merge-shards to combine outputs afterwards.
- Parallel: --workers N enables multiprocessing on a single node.
- No unnecessary disk output: no per-image PNGs or QC overlays by default.

Output
------
  <output_dir>/
    iqms.csv              one row per (image × variant); incrementally written
    completed.txt         image_ids that finished successfully (for resume)
    errors.csv            images or variants that failed
    run_summary.json      final statistics

  For sharded runs, each shard writes into <output_dir>/shard_{K:04d}/.
  Use --merge-shards to merge them into <output_dir>/iqms_merged.csv.

Example commands
----------------
Full dataset, single node, 8 parallel workers:
    python compute_iqms_full_dataset.py --workers 8 --output-dir ./iqm_results

SLURM job array (100 shards):
    python compute_iqms_full_dataset.py \\
        --n-shards 100 --shard-index $SLURM_ARRAY_TASK_ID \\
        --output-dir ./iqm_results

Merge all shard CSVs into one:
    python compute_iqms_full_dataset.py --merge-shards \\
        --n-shards 100 --output-dir ./iqm_results
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import importlib.util
import json
import logging
import sys
import time
import threading
import warnings
from pathlib import Path
from typing import Any

import numpy as np

# Suppress pydicom's VR validation warning: the VinDr-Mammo DICOMs contain
# non-standard UID strings that are harmless for pixel data processing.
warnings.filterwarnings(
    "ignore",
    message="Invalid value for VR UI",
    category=UserWarning,
    module="pydicom",
)


# =============================================================================
# Paths: update if your dataset is mounted elsewhere
# =============================================================================

VINDR_ROOT = Path(
    r"D:\Mammo\vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0"
)
IMAGES_DIR   = VINDR_ROOT / "images"
FINDING_CSV  = VINDR_ROOT / "finding_annotations.csv"
BREAST_CSV   = VINDR_ROOT / "breast-level_annotations.csv"
HERE         = Path(__file__).resolve().parent
TOOLKIT_PATH = HERE / "roi_metrics.py"


# =============================================================================
# Output column schema: every CSV row has exactly these columns in this order
# =============================================================================

COLUMNS: list[str] = [
    # Core metadata
    "image_id", "study_id", "split", "has_lesion_annotation",
    "degradation_type", "severity", "parameter_name", "parameter_value",
    # Metrics — all images
    "noise_variance", "tenengrad", "ssim",
    # Metrics — lesion images only (NaN for others)
    "delta_mu", "cnr",
    "cnr_lesion_mean", "cnr_bg_mean", "cnr_bg_std", "cnr_bg_pixels",
    # ROI / processing status
    "breast_mask_valid", "tissue_patch_valid", "lesion_roi_valid",
    "bg_ring_valid", "normalization_valid",
    "processing_status", "error_message",
    # Optional geometry
    "pixel_spacing_mm", "breast_area_pixels", "crop_height", "crop_width",
]

_METRIC_KEYS = (
    "noise_variance", "tenengrad", "ssim",
    "delta_mu", "cnr",
    "cnr_lesion_mean", "cnr_bg_mean", "cnr_bg_std", "cnr_bg_pixels",
)

_NAN_METRICS: dict[str, Any] = {k: "" for k in _METRIC_KEYS}


# =============================================================================
# Module-level toolkit reference: populated by _worker_init in subprocesses
# or set directly in single-process mode
# =============================================================================

_toolkit = None  # type: ignore[assignment]


def _load_toolkit(path: Path):
    spec = importlib.util.spec_from_file_location("roi_metrics", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load toolkit from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _worker_init(toolkit_path_str: str) -> None:
    """Called once per worker process by ProcessPoolExecutor."""
    global _toolkit
    _toolkit = _load_toolkit(Path(toolkit_path_str))


# =============================================================================
# Annotation index
# =============================================================================

def build_annotation_index(
    finding_csv: Path,
    breast_csv: Path,
) -> dict[str, dict[str, Any]]:
    """Return {image_id: {study_id, split, has_lesion_annotation, lesion_bboxes_full}}.

    Iterates both CSVs once and builds an in-memory lookup so that per-image
    annotation queries are O(1) rather than O(n) during the processing loop.
    """
    index: dict[str, dict] = {}

    def _ensure(iid: str, row: dict) -> None:
        if iid not in index:
            index[iid] = {
                "study_id": row.get("study_id", "").strip(),
                "split": row.get("split", "").strip(),
                "has_lesion_annotation": False,
                "lesion_bboxes_full": [],
            }
        else:
            if not index[iid]["study_id"]:
                index[iid]["study_id"] = row.get("study_id", "").strip()
            if not index[iid]["split"]:
                index[iid]["split"] = row.get("split", "").strip()

    # Breast-level CSV covers every image (including "No Finding" images)
    if breast_csv.exists():
        with breast_csv.open("r", newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                iid = row.get("image_id", "").strip()
                if iid:
                    _ensure(iid, row)
        logging.info("Loaded breast-level metadata for %d images.", len(index))
    else:
        logging.warning("breast-level_annotations.csv not found: %s", breast_csv)

    if finding_csv.exists():
        n_lesion_rows = 0
        with finding_csv.open("r", newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                iid = row.get("image_id", "").strip()
                if not iid:
                    continue
                _ensure(iid, row)

                finding = row.get("finding_categories", "")
                xmin_s = row.get("xmin", "").strip()
                if "No Finding" in finding:
                    continue
                if not xmin_s or xmin_s.lower() in ("nan", "none", ""):
                    continue
                try:
                    x1 = float(row["xmin"])
                    y1 = float(row["ymin"])
                    x2 = float(row["xmax"])
                    y2 = float(row["ymax"])
                    if not (x2 > x1 and y2 > y1):
                        continue
                    bbox = (int(round(x1)), int(round(y1)),
                            int(round(x2)), int(round(y2)))
                except (ValueError, TypeError, KeyError):
                    continue

                index[iid]["has_lesion_annotation"] = True
                index[iid]["lesion_bboxes_full"].append(bbox)
                n_lesion_rows += 1

        n_lesion_images = sum(1 for v in index.values() if v["has_lesion_annotation"])
        logging.info(
            "Loaded %d lesion bbox rows for %d distinct images.",
            n_lesion_rows, n_lesion_images,
        )
    else:
        logging.warning("finding_annotations.csv not found: %s", finding_csv)

    return index


# =============================================================================
# DICOM discovery
# =============================================================================

def discover_dicoms(images_dir: Path) -> list[Path]:
    """Find all DICOM files under images_dir in deterministic sorted order."""
    patterns = ["*.dicom", "*.dcm", "*.DICOM", "*.DCM"]
    seen: set[str] = set()
    found: list[Path] = []
    for pat in patterns:
        for p in images_dir.rglob(pat):
            key = str(p).lower()
            if key not in seen:
                seen.add(key)
                found.append(p)
    found.sort()  # deterministic order → stable per-image seeds
    return found


# =============================================================================
# Per-image processing: must be at module level to be picklable
# =============================================================================

def _param_info(spec_dict: dict) -> tuple[str, Any]:
    """Return (parameter_name, parameter_value) for a degradation spec dict."""
    dt = spec_dict["degradation_type"]
    params = spec_dict["params"]
    if dt == "none":
        return "none", ""
    if dt == "noise":
        return "dose_factor", params.get("dose_factor", "")
    if dt == "motion_blur":
        return "length_mm", params.get("length_mm", "")
    if dt == "contrast":
        return "alpha", params.get("alpha", "")
    if dt == "jpeg2000":
        return "compression_ratio", params.get("compression_ratio", "")
    return "unknown", ""


def _clean(v: Any) -> Any:
    """Convert NaN and None to empty string for cleaner CSV output."""
    if v is None:
        return ""
    if isinstance(v, float) and v != v:  # NaN check (NaN != NaN)
        return ""
    return v


def _process_image(
    image_path_str: str,
    image_meta: dict,
    cfg: dict,
    plan_specs: list[dict],
) -> tuple[list[dict], str]:
    """Process one image and return (rows, status).

    This function is module-level so that ProcessPoolExecutor can pickle it.
    The global _toolkit must be populated before calling this (either via
    _worker_init in a subprocess, or manually in single-process mode).

    Parameters
    ----------
    image_path_str : str
        Absolute path to the DICOM file.
    image_meta : dict
        {study_id, split, has_lesion_annotation, lesion_bboxes_full}
    cfg : dict
        {patch_size, buffer_px, base_seed}  (base_seed is image-specific)
    plan_specs : list[dict]
        Serialised DegradationSpec objects:
        [{degradation_type, severity, params}, ...]

    Returns
    -------
    rows : list[dict]
        One row per variant (19 rows for the default plan).
        On a complete image failure, a single error row is returned.
    status : str
        'ok' or 'failed'
    """
    global _toolkit
    tk = _toolkit
    if tk is None:
        raise RuntimeError(
            "_toolkit is None — call _worker_init or set _toolkit before calling _process_image."
        )

    image_path = Path(image_path_str)
    image_id   = image_path.stem
    # study_id from annotation index; fall back to parent directory name
    # (VinDr-Mammo layout: images/<study_id>/<image_id>.dicom)
    study_id   = image_meta.get("study_id", "") or image_path.parent.name
    split      = image_meta.get("split", "")
    has_lesion = bool(image_meta.get("has_lesion_annotation", False))
    lesion_bboxes_full: list[tuple] = [
        tuple(b) for b in image_meta.get("lesion_bboxes_full", [])
    ]

    patch_size: int = cfg["patch_size"]
    buffer_px:  int = cfg["buffer_px"]
    base_seed:  int = cfg["base_seed"]

    # ------------------------------------------------------------------ load
    try:
        load_res = tk.read_mammogram(image_path)
    except Exception as e:
        return _error_row(image_id, study_id, split, has_lesion, e), "failed"

    pixel_spacing_mm: Any = ""
    if load_res.metadata:
        ps = load_res.metadata.get("PixelSpacing_mm")
        if ps is not None:
            pixel_spacing_mm = ps

    raw_image = load_res.image

    # ------------------------------------------------------------- breast mask
    try:
        breast_mask_full = tk.create_breast_mask(raw_image)
        breast_mask_valid = bool(np.any(breast_mask_full))
    except Exception as e:
        return _error_row(image_id, study_id, split, has_lesion, e), "failed"

    # -------------------------------------------------------------------- crop
    try:
        crop_res  = tk.crop_to_breast(raw_image, breast_mask_full, buffer_px=buffer_px)
        raw_crop  = crop_res.image
        mask_crop = crop_res.mask.astype(bool)
        crop_h    = int(raw_crop.shape[0])
        crop_w    = int(raw_crop.shape[1])
        breast_area_px = int(mask_crop.sum())
    except Exception as e:
        return _error_row(image_id, study_id, split, has_lesion, e), "failed"

    # ---------------------------------------------------- shift lesion bboxes
    lesion_bboxes_crop: list = []
    if lesion_bboxes_full:
        try:
            lesion_bboxes_crop = tk.shift_bboxes_to_crop(lesion_bboxes_full, crop_res.bbox)
        except Exception:
            lesion_bboxes_crop = []

    primary_lesion = tk.select_primary_lesion_bbox(lesion_bboxes_crop)

    # ----------------------------------------------------- fixed normalization
    try:
        norm_params    = tk.estimate_fixed_normalization(raw_crop, mask=mask_crop)
        original_crop  = tk.apply_fixed_normalization(raw_crop, norm_params)
        normalization_valid = True
    except Exception as e:
        return _error_row(image_id, study_id, split, has_lesion, e), "failed"

    # ------------------------------------------------------- fixed tissue patch
    tissue_patch_valid  = False
    baseline_noise_std  = 0.0
    patch_res           = None
    try:
        patch_res = tk.select_homogeneous_tissue_patch(
            original_crop, mask_crop,
            lesion_bboxes=lesion_bboxes_crop,
            patch_size=patch_size,
        )
        tissue_patch_valid = True
        baseline_noise_std = float(np.std(patch_res.patch))
    except Exception:
        # Patch selection failed — noise_variance will be NaN but we continue
        tissue_patch_valid = False

    # ---------------------------------------------------------- lesion ring
    ring_mask      = None
    lesion_roi_valid = primary_lesion is not None
    bg_ring_valid    = False

    if primary_lesion is not None:
        try:
            _, ring_mask, _ = tk.lesion_background_ring(
                primary_lesion,
                original_crop.shape,
                mask_crop,
                other_lesions=lesion_bboxes_crop,
            )
            if ring_mask is not None and np.any(ring_mask):
                bg_ring_valid = True
            else:
                ring_mask = None
        except Exception:
            ring_mask = None

    # ---------------------------------------------------- per-variant metrics
    rows: list[dict] = []

    for spec_idx, spec_dict in enumerate(plan_specs):
        seed = base_seed + spec_idx
        param_name, param_value = _param_info(spec_dict)
        spec = tk.DegradationSpec(
            spec_dict["degradation_type"],
            spec_dict["severity"],
            spec_dict["params"],
        )

        variant_status = "ok"
        error_msg      = ""
        metrics: dict[str, Any] = dict(_NAN_METRICS)

        try:
            degraded = tk.apply_degradation(
                image=original_crop,
                degradation_type=spec.degradation_type,
                severity=spec.severity,
                pixel_spacing_mm=load_res.metadata.get("PixelSpacing_mm") if load_res.metadata else None,
                baseline_noise_std=baseline_noise_std,
                seed=seed,
                mask=mask_crop,
            )
            computed = tk.compute_fixed_roi_metrics(
                original_img=original_crop,
                degraded_img=degraded,
                breast_mask=mask_crop,
                tissue_patch_bbox=patch_res.bbox if patch_res is not None else (0, 0, 1, 1),
                lesion_bbox=primary_lesion,
                ring_mask=ring_mask,
            )
            for k in _METRIC_KEYS:
                metrics[k] = _clean(computed.get(k))
        except Exception as ve:
            variant_status = "variant_failed"
            error_msg      = str(ve)[:500]

        row: dict[str, Any] = {
            "image_id":              image_id,
            "study_id":              study_id,
            "split":                 split,
            "has_lesion_annotation": int(has_lesion),
            "degradation_type":      spec.degradation_type,
            "severity":              int(spec.severity),
            "parameter_name":        param_name,
            "parameter_value":       param_value,
            **metrics,
            "breast_mask_valid":     int(breast_mask_valid),
            "tissue_patch_valid":    int(tissue_patch_valid),
            "lesion_roi_valid":      int(lesion_roi_valid),
            "bg_ring_valid":         int(bg_ring_valid),
            "normalization_valid":   int(normalization_valid),
            "processing_status":     variant_status,
            "error_message":         error_msg,
            "pixel_spacing_mm":      pixel_spacing_mm,
            "breast_area_pixels":    breast_area_px,
            "crop_height":           crop_h,
            "crop_width":            crop_w,
        }
        rows.append(row)

    return rows, "ok"


def _error_row(
    image_id: str,
    study_id: str,
    split: str,
    has_lesion: bool,
    exc: Exception,
) -> list[dict]:
    """Return a single-row list representing a completely failed image."""
    return [{
        "image_id":              image_id,
        "study_id":              study_id,
        "split":                 split,
        "has_lesion_annotation": int(has_lesion),
        "degradation_type":      "ERROR",
        "severity":              -1,
        "parameter_name":        "",
        "parameter_value":       "",
        **_NAN_METRICS,
        "breast_mask_valid":     0,
        "tissue_patch_valid":    0,
        "lesion_roi_valid":      0,
        "bg_ring_valid":         0,
        "normalization_valid":   0,
        "processing_status":     "failed",
        "error_message":         str(exc)[:500],
        "pixel_spacing_mm":      "",
        "breast_area_pixels":    "",
        "crop_height":           "",
        "crop_width":            "",
    }]


# =============================================================================
# CSV helpers
# =============================================================================

def _csv_needs_header(csv_path: Path) -> bool:
    """Return True if the CSV file does not yet have a header row."""
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return True
    with csv_path.open("r", encoding="utf-8") as f:
        first = f.readline().strip()
    return first != ",".join(COLUMNS)


def append_rows_to_csv(rows: list[dict], csv_path: Path, lock: threading.Lock) -> None:
    """Append a list of row dicts to the CSV, writing the header if absent."""
    with lock:
        write_header = _csv_needs_header(csv_path)
        with csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=COLUMNS,
                extrasaction="ignore",
                lineterminator="\n",
            )
            if write_header:
                writer.writeheader()
            for row in rows:
                writer.writerow({col: row.get(col, "") for col in COLUMNS})


def load_completed_ids(completed_path: Path) -> set[str]:
    """Read completed image IDs from completed.txt."""
    if not completed_path.exists():
        return set()
    with completed_path.open("r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def append_completed_id(image_id: str, completed_path: Path, lock: threading.Lock) -> None:
    """Atomically append one image_id to completed.txt."""
    with lock:
        with completed_path.open("a", encoding="utf-8") as f:
            f.write(image_id + "\n")


# =============================================================================
# Main processing loop
# =============================================================================

def run_full_dataset(
    all_dicoms: list[Path],
    annotation_index: dict,
    out_dir: Path,
    toolkit_path: Path,
    patch_size: int,
    buffer_px: int,
    base_seed: int,
    workers: int,
    plan_specs: list[dict],
    shard_index: int | None,
    n_shards: int | None,
) -> dict:
    """Iterate over (a shard of) the dataset and write metrics incrementally.

    Returns a summary dict.
    """
    # ---------------------------------------------------------------- sharding
    if shard_index is not None and n_shards is not None and n_shards > 1:
        # Interleaved sharding: shard K gets indices K, K+N, K+2N, ...
        # This distributes large and small files evenly across shards.
        shard_dicoms = [p for i, p in enumerate(all_dicoms) if i % n_shards == shard_index]
        shard_tag = f"shard_{shard_index:04d}"
        shard_out_dir = out_dir / shard_tag
        logging.info(
            "Shard %d/%d: %d / %d images.",
            shard_index, n_shards, len(shard_dicoms), len(all_dicoms),
        )
    else:
        shard_dicoms = all_dicoms
        shard_out_dir = out_dir
        shard_tag = "full"

    shard_out_dir.mkdir(parents=True, exist_ok=True)

    csv_path       = shard_out_dir / "iqms.csv"
    completed_path = shard_out_dir / "completed.txt"
    errors_path    = shard_out_dir / "errors.csv"

    # ----------------------------------------------------------- resume state
    completed_ids = load_completed_ids(completed_path)
    todo = [p for p in shard_dicoms if p.stem not in completed_ids]
    skipped = len(shard_dicoms) - len(todo)
    if skipped > 0:
        logging.info("Resuming: skipping %d already-completed images.", skipped)
    logging.info("Images to process: %d", len(todo))

    # --------------------------------------------------------------- counters
    n_ok          = 0
    n_failed      = 0
    n_variant_ok  = 0
    n_variant_err = 0
    n_lesion_rows = 0
    t0            = time.time()

    # ------------------------------------------------------------------ locks
    csv_lock  = threading.Lock()
    err_lock  = threading.Lock()
    done_lock = threading.Lock()

    def _handle_result(image_path: Path, rows: list[dict], status: str) -> None:
        nonlocal n_ok, n_failed, n_variant_ok, n_variant_err, n_lesion_rows
        if status == "ok":
            n_ok += 1
            append_completed_id(image_path.stem, completed_path, done_lock)
        else:
            n_failed += 1
            append_rows_to_csv(rows, errors_path, err_lock)

        for r in rows:
            if r.get("processing_status") in ("ok", "variant_failed"):
                n_variant_ok  += (r.get("processing_status") == "ok")
                n_variant_err += (r.get("processing_status") == "variant_failed")
            if r.get("cnr", "") != "":
                n_lesion_rows += 1

        append_rows_to_csv(rows, csv_path, csv_lock)

    # --------------------------------------------------------- process images
    if workers <= 1:
        # ---- Single-process mode ----------------------------------------
        global _toolkit
        _toolkit = _load_toolkit(toolkit_path)

        for idx, image_path in enumerate(todo):
            image_id = image_path.stem
            meta     = annotation_index.get(image_id, {})
            cfg      = {"patch_size": patch_size, "buffer_px": buffer_px,
                        "base_seed": base_seed + _stable_index(image_path, all_dicoms) * 1000}

            t_img = time.time()
            try:
                rows, status = _process_image(str(image_path), meta, cfg, plan_specs)
            except Exception as e:
                rows, status = _error_row(image_id,
                                          meta.get("study_id", "") or image_path.parent.name,
                                          meta.get("split", ""), bool(meta.get("has_lesion_annotation")),
                                          e), "failed"
            _handle_result(image_path, rows, status)

            elapsed  = time.time() - t_img
            total_el = time.time() - t0
            avg_s    = total_el / (idx + 1)
            # ETA for the remaining images in this shard/run
            remain_s = avg_s * (len(todo) - idx - 1)
            # ETA extrapolated to the full dataset (useful when sharding)
            full_remain_s = avg_s * (len(all_dicoms) - skipped - (idx + 1))
            err_hint = ""
            if status != "ok" and rows:
                err_hint = f" — {rows[0].get('error_message', '')[:80]}"
            logging.info(
                "[%d/%d] %s | %s | %.1fs/img | ETA this run: %.0fm | full dataset: %.0fm%s",
                idx + 1, len(todo), image_id,
                "OK" if status == "ok" else "FAILED",
                avg_s, remain_s / 60, full_remain_s / 60, err_hint,
            )

    else:
        # ---- Multiprocessing mode ---------------------------------------
        logging.info("Starting %d worker processes.", workers)
        args_list = []
        for image_path in todo:
            image_id = image_path.stem
            meta     = annotation_index.get(image_id, {})
            cfg      = {"patch_size": patch_size, "buffer_px": buffer_px,
                        "base_seed": base_seed + _stable_index(image_path, all_dicoms) * 1000}
            args_list.append((str(image_path), meta, cfg, plan_specs))

        n_done = 0
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers,
            initializer=_worker_init,
            initargs=(str(toolkit_path),),
        ) as executor:
            futures = {
                executor.submit(_process_image, *args): args[0]
                for args in args_list
            }
            for future in concurrent.futures.as_completed(futures):
                image_path_str = futures[future]
                image_path     = Path(image_path_str)
                n_done        += 1
                try:
                    rows, status = future.result()
                except Exception as e:
                    image_id = image_path.stem
                    meta     = annotation_index.get(image_id, {})
                    rows     = _error_row(
                        image_id,
                        meta.get("study_id", "") or image_path.parent.name,
                        meta.get("split", ""),
                        bool(meta.get("has_lesion_annotation")),
                        e,
                    )
                    status = "failed"

                _handle_result(image_path, rows, status)
                total_el  = time.time() - t0
                avg_s     = total_el / n_done
                remain_s  = avg_s * (len(todo) - n_done)
                full_remain_s = avg_s * (len(all_dicoms) - skipped - n_done)
                err_hint  = ""
                if status != "ok" and rows:
                    err_hint = f" — {rows[0].get('error_message', '')[:80]}"
                logging.info(
                    "[%d/%d] %s | %s | %.1fs/img | ETA this run: %.0fm | full dataset: %.0fm%s",
                    n_done, len(todo), image_path.name,
                    "OK" if status == "ok" else "FAILED",
                    avg_s, remain_s / 60, full_remain_s / 60, err_hint,
                )

    # --------------------------------------------------------------- summary
    total_time = time.time() - t0
    summary = {
        "shard":                  shard_tag,
        "output_dir":             str(shard_out_dir),
        "images_in_shard":        len(shard_dicoms),
        "images_skipped_resume":  skipped,
        "images_processed":       n_ok + n_failed,
        "images_ok":              n_ok,
        "images_failed":          n_failed,
        "variant_rows_ok":        n_variant_ok,
        "variant_rows_failed":    n_variant_err,
        "lesion_metric_rows":     n_lesion_rows,
        "total_rows_written":     n_variant_ok + n_variant_err,
        "total_time_s":           round(total_time, 1),
        "avg_s_per_image":        round(total_time / max(1, n_ok + n_failed), 2),
        "csv_path":               str(csv_path),
        "completed_path":         str(completed_path),
    }

    _print_summary(summary)

    with (shard_out_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def _stable_index(image_path: Path, all_dicoms: list[Path]) -> int:
    """Return the stable sorted index of image_path in the full file list.

    This ensures that each image always gets the same per-image seed,
    regardless of which shard or worker processes it.
    """
    # Binary search on sorted list (all_dicoms is sorted in discover_dicoms)
    target = str(image_path)
    lo, hi = 0, len(all_dicoms) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        s = str(all_dicoms[mid])
        if s == target:
            return mid
        elif s < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return hash(target) % (2 ** 20)  # fallback (should not happen)


def _print_summary(s: dict) -> None:
    print()
    print("=" * 64)
    print("  IQM COMPUTATION SUMMARY")
    print(f"  Shard:              {s['shard']}")
    print(f"  Images processed:   {s['images_ok']} ok / {s['images_failed']} failed")
    print(f"  Resume-skipped:     {s['images_skipped_resume']}")
    print(f"  Variant rows ok:    {s['variant_rows_ok']}")
    print(f"  Variant rows err:   {s['variant_rows_failed']}")
    print(f"  Lesion metric rows: {s['lesion_metric_rows']}")
    print(f"  Total rows written: {s['total_rows_written']}")
    print(f"  Total time:         {s['total_time_s']:.0f}s "
          f"({s['avg_s_per_image']:.2f}s/image)")
    print(f"  Output:             {s['output_dir']}")
    print(f"  CSV:                {s['csv_path']}")
    print("=" * 64)


# =============================================================================
# Shard merge
# =============================================================================

def merge_shards(out_dir: Path, n_shards: int) -> Path:
    """Concatenate all shard CSVs into one iqms_merged.csv.

    The header is taken from the first shard CSV that has one; subsequent
    shard files have their header row stripped before appending.
    """
    merged_path = out_dir / "iqms_merged.csv"
    total_rows  = 0
    header_written = False

    with merged_path.open("w", newline="", encoding="utf-8") as out_f:
        for shard_idx in range(n_shards):
            shard_csv = out_dir / f"shard_{shard_idx:04d}" / "iqms.csv"
            if not shard_csv.exists():
                logging.warning("Shard CSV not found: %s", shard_csv)
                continue
            with shard_csv.open("r", newline="", encoding="utf-8") as in_f:
                reader = csv.reader(in_f)
                for i, row in enumerate(reader):
                    if i == 0:  # header
                        if not header_written:
                            out_f.write(",".join(row) + "\n")
                            header_written = True
                        continue
                    out_f.write(",".join(row) + "\n")
                    total_rows += 1

    logging.info("Merged %d shards → %d rows → %s", n_shards, total_rows, merged_path)
    print(f"\nMerged {n_shards} shards: {total_rows} rows → {merged_path}")
    return merged_path


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Full-dataset IQM computation for VinDr-Mammo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # -- Data paths
    p.add_argument("--images-dir",   type=Path, default=IMAGES_DIR,
                   help="Root directory with study sub-folders of DICOMs")
    p.add_argument("--finding-csv",  type=Path, default=FINDING_CSV,
                   help="Path to finding_annotations.csv")
    p.add_argument("--breast-csv",   type=Path, default=BREAST_CSV,
                   help="Path to breast-level_annotations.csv")
    p.add_argument("--toolkit",      type=Path, default=TOOLKIT_PATH,
                   help="Path to roi_metrics.py (default: same directory as this script)")

    # -- Output
    p.add_argument("--output-dir",   type=Path, default=HERE / "iqm_full_dataset",
                   help="Output directory (default: ./iqm_full_dataset)")

    # -- Processing parameters
    p.add_argument("--patch-size",   type=int,  default=96,
                   help="Homogeneous tissue patch size in pixels (default: 96)")
    p.add_argument("--buffer-px",    type=int,  default=20,
                   help="Breast crop buffer in pixels (default: 20)")
    p.add_argument("--seed",         type=int,  default=42,
                   help="Global base seed; per-image seeds are derived from this")

    # -- Parallelism
    p.add_argument("--workers",      type=int,  default=1,
                   help="Number of parallel worker processes (default: 1 = single-process)")

    # -- Sharding (SLURM job arrays)
    p.add_argument("--shard-index",  type=int,  default=None,
                   help="0-based index of this shard (set to $SLURM_ARRAY_TASK_ID)")
    p.add_argument("--n-shards",     type=int,  default=None,
                   help="Total number of shards for this job array")

    # -- Merge mode
    p.add_argument("--merge-shards", action="store_true",
                   help="Merge all shard CSVs into iqms_merged.csv and exit")

    # -- Logging
    p.add_argument("--log-level",    default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                   help="Logging verbosity (default: INFO)")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # ------------------------------------------------------------------ merge
    if args.merge_shards:
        if args.n_shards is None:
            print("ERROR: --merge-shards requires --n-shards N")
            sys.exit(1)
        merge_shards(args.output_dir, args.n_shards)
        return

    # ---------------------------------------------------------- discover DICOMs
    logging.info("Scanning for DICOMs in: %s", args.images_dir)
    all_dicoms = discover_dicoms(args.images_dir)
    if not all_dicoms:
        logging.error("No DICOM files found. Check --images-dir.")
        sys.exit(1)
    logging.info("Found %d DICOM files.", len(all_dicoms))

    # --------------------------------------------------- annotation index
    logging.info("Building annotation index …")
    annotation_index = build_annotation_index(args.finding_csv, args.breast_csv)
    logging.info("Annotation index: %d image entries.", len(annotation_index))

    # --------------------------------------------------- degradation plan
    # Load toolkit once in the main process to build the plan, then serialise
    # it as plain dicts so workers can reconstruct without needing the same
    # module instance.
    logging.info("Loading toolkit from: %s", args.toolkit)
    tk = _load_toolkit(args.toolkit)
    plan = tk.build_default_degradation_plan()
    plan_specs = [
        {"degradation_type": s.degradation_type, "severity": s.severity, "params": dict(s.params)}
        for s in plan
    ]
    logging.info("Degradation plan: %d variants per image (1 baseline + %d degradations: 6 noise + 6 motion_blur + 6 contrast + 6 jpeg2000).",
                 len(plan), len(plan) - 1)
    logging.info("Total metric computations (full dataset): %d × %d = %d",
                 len(all_dicoms), len(plan), len(all_dicoms) * len(plan))

    # Estimate runtime (rough: assume 5s per image on a modern CPU)
    est_minutes = len(all_dicoms) * 5 / max(1, args.workers) / 60
    logging.info("Estimated runtime: ~%.0f minutes with %d worker(s).", est_minutes, args.workers)

    # --------------------------------------------------- run
    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_full_dataset(
        all_dicoms=all_dicoms,
        annotation_index=annotation_index,
        out_dir=args.output_dir,
        toolkit_path=args.toolkit,
        patch_size=args.patch_size,
        buffer_px=args.buffer_px,
        base_seed=args.seed,
        workers=args.workers,
        plan_specs=plan_specs,
        shard_index=args.shard_index,
        n_shards=args.n_shards,
    )


if __name__ == "__main__":
    # Required for multiprocessing on Windows (spawn start method)
    import multiprocessing
    multiprocessing.freeze_support()
    main()
