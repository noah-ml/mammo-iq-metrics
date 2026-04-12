from __future__ import annotations

"""
DICOM-first experiment runner for VinDr-Mammo ROI + IQ metrics.

Capabilities
------------
- Single-image mode via --image
- Batch mode via --image-list (TXT, CSV, or TSV)
- DICOM-first loading with PNG/JPG fallback
- Breast mask / crop on the original image
- Fixed normalization estimated once on the original crop
- Fixed ROIs (tissue patch + lesion ring) determined once on the original crop
- Baseline + controlled degradations using the same ROIs
- Per-image QC PNGs + per-run CSV/JSON summaries
"""

import argparse
import csv
import importlib.util
import json
from pathlib import Path
from typing import Iterable
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
TOOLKIT_PATH = HERE / "roi_metrics_dicom_3.0.py"


def load_toolkit(toolkit_path: Path):
    spec = importlib.util.spec_from_file_location("roi_metrics_dicom_3.0", toolkit_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load toolkit from: {toolkit_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _normalize_name(x: str) -> str:
    return x.strip().lower().replace(" ", "_")


def _find_first_key(d: dict, candidates: Iterable[str]) -> str | None:
    keys = {_normalize_name(k): k for k in d.keys()}
    for c in candidates:
        if c in keys:
            return keys[c]
    return None


def load_bboxes_from_csv(csv_path: Path, image_identifier: str) -> list[tuple[int, int, int, int]]:
    rows: list[tuple[int, int, int, int]] = []
    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)
        if not all_rows:
            return rows

        id_key = _find_first_key(
            all_rows[0],
            ["image_id", "study_id", "image_name", "file_name", "filename", "path", "sop_instance_uid"],
        )
        if id_key is None:
            raise ValueError(
                "Could not find image id column in annotation CSV. "
                "Expected one of: image_id, study_id, image_name, file_name, filename, path, sop_instance_uid"
            )

        x1_key = _find_first_key(all_rows[0], ["x_min", "xmin", "x1"])
        y1_key = _find_first_key(all_rows[0], ["y_min", "ymin", "y1"])
        x2_key = _find_first_key(all_rows[0], ["x_max", "xmax", "x2"])
        y2_key = _find_first_key(all_rows[0], ["y_max", "ymax", "y2"])
        if None in [x1_key, y1_key, x2_key, y2_key]:
            return rows  # no bbox columns -> treat as no lesion annotations

        image_identifier_l = image_identifier.lower()
        stem_l = Path(image_identifier).stem.lower()

        for row in all_rows:
            raw_id = str(row[id_key]).strip()
            raw_id_l = raw_id.lower()
            raw_stem_l = Path(raw_id).stem.lower()
            if raw_id_l != image_identifier_l and raw_stem_l != stem_l:
                continue

            try:
                x1 = float(row[x1_key])
                y1 = float(row[y1_key])
                x2 = float(row[x2_key])
                y2 = float(row[y2_key])
            except Exception:
                continue

            if not np.isfinite([x1, y1, x2, y2]).all():
                continue
            if x2 <= x1 or y2 <= y1:
                continue

            rows.append((
                int(round(x1)),
                int(round(y1)),
                int(round(x2)),
                int(round(y2)),
            ))
    return rows


def read_image_list(path: Path) -> list[Path]:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".lst"}:
        with path.open("r", encoding="utf-8") as f:
            return [Path(line.strip()) for line in f if line.strip()]

    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open("r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            rows = list(reader)
            if not rows:
                return []
            key = _find_first_key(rows[0], ["image_path", "path", "file", "filename"])
            if key is None:
                raise ValueError("Could not find image path column in image list file.")
            return [Path(str(row[key]).strip()) for row in rows if str(row[key]).strip()]

    raise ValueError("Unsupported image list format. Use TXT, CSV, or TSV.")


def save_metrics_csv(rows: list[dict], out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["status"])
            writer.writerow(["no_rows"])
        return

    fieldnames = sorted({k for row in rows for k in row.keys()})
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_single_degradation_spec(toolkit, degradation: str, severity: int = 1):
    """Build a minimal [baseline, one-degradation] plan for ad-hoc CLI use.

    ``severity`` maps to the same SEVERITY_PRESETS used by the full plan (1–6).
    """
    if degradation == "none":
        return [toolkit.DegradationSpec("none", 0, {})]
    if degradation in ("noise", "motion_blur", "contrast"):
        return [
            toolkit.DegradationSpec("none", 0, {}),
            toolkit.DegradationSpec(degradation, int(severity), {}),
        ]
    raise ValueError(
        f"Unknown degradation '{degradation}'. "
        "Choose from: none, noise, motion_blur, contrast"
    )


def parse_args():
    p = argparse.ArgumentParser(description="Run fixed-ROI IQ experiments for VinDr-Mammo")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="Path to one mammogram image (DICOM preferred; PNG/JPG fallback)")
    src.add_argument("--image-list", help="TXT/CSV/TSV containing image paths for batch mode")

    p.add_argument("--annotations", default=None, help="Optional CSV with lesion bounding boxes")
    p.add_argument("--image-id", default=None, help="Optional image identifier for single-image mode")
    p.add_argument("--output-dir", required=True, help="Directory for metrics and QC outputs")

    p.add_argument("--patch-size", type=int, default=96, help="Homogeneous tissue patch size")
    p.add_argument("--buffer-px", type=int, default=20, help="Breast crop buffer in pixels")
    p.add_argument("--seed", type=int, default=1234, help="Base random seed")
    p.add_argument("--save-degraded-qc", action="store_true", help="Save QC overlay for each degraded variant")
    p.add_argument("--dicom-force-invert", action="store_true", help="Force inversion for DICOM images")

    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--full-plan", action="store_true", help="Run baseline + default 12 degradations (default)")
    mode.add_argument(
        "--degradation",
        choices=["none", "noise", "motion_blur", "contrast"],
        default=None,
        help="Run only baseline + one custom degradation",
    )
    p.add_argument("--deg-severity", type=int, default=1, choices=range(1, 7),
                   metavar="1-6", help="Severity level for --degradation (default: 1)")
    return p.parse_args()


def process_one_image(
    image_path: Path,
    toolkit,
    out_dir: Path,
    annotations_csv: Path | None,
    image_id_override: str | None,
    patch_size: int,
    buffer_px: int,
    base_seed: int,
    save_degraded_qc: bool,
    plan,
    dicom_force_invert: bool,
) -> tuple[list[dict], dict]:
    load_res = toolkit.read_mammogram(image_path, dicom_force_invert=True if dicom_force_invert else None)
    raw_image = load_res.image

    # Extract pixel spacing for physically calibrated motion blur.
    # Falls back to None if not present (PNG/JPG input or missing DICOM tag),
    # in which case apply_degradation uses the pre-computed pixel fallback.
    pixel_spacing_mm: float | None = None
    if load_res.metadata is not None:
        pixel_spacing_mm = load_res.metadata.get("PixelSpacing_mm")

    image_id = image_id_override or image_path.stem
    lesion_bboxes_full: list[tuple[int, int, int, int]] = []
    if annotations_csv is not None:
        lesion_bboxes_full = load_bboxes_from_csv(annotations_csv, image_id)
        if not lesion_bboxes_full:
            sop_uid = None
            if load_res.metadata is not None:
                sop_uid = load_res.metadata.get("SOPInstanceUID")
            if sop_uid:
                lesion_bboxes_full = load_bboxes_from_csv(annotations_csv, str(sop_uid))

    breast_mask_full = toolkit.create_breast_mask(raw_image)
    crop_res = toolkit.crop_to_breast(raw_image, breast_mask_full, buffer_px=buffer_px)
    raw_crop = crop_res.image
    mask_crop = crop_res.mask.astype(bool)

    lesion_bboxes_crop = toolkit.shift_bboxes_to_crop(lesion_bboxes_full, crop_res.bbox) if lesion_bboxes_full else []
    primary_lesion = toolkit.select_primary_lesion_bbox(lesion_bboxes_crop)

    norm_params = toolkit.estimate_fixed_normalization(raw_crop, mask=mask_crop)
    original_crop = toolkit.apply_fixed_normalization(raw_crop, norm_params)

    patch_res = toolkit.select_homogeneous_tissue_patch(
        original_crop,
        mask_crop,
        lesion_bboxes=lesion_bboxes_crop,
        patch_size=patch_size,
    )

    # Baseline noise std estimated from the original tissue patch in [0,1] space.
    # Used as the reference for dose-based noise scaling: a physically lower
    # dose is simulated by adding noise on top of this baseline level.
    baseline_noise_std = float(np.std(patch_res.patch))

    ring_mask = None
    ring_outer_bbox = None
    if primary_lesion is not None:
        _, ring_mask, ring_outer_bbox = toolkit.lesion_background_ring(
            primary_lesion,
            original_crop.shape,
            mask_crop,
            other_lesions=lesion_bboxes_crop,
        )
        if ring_mask is not None and not np.any(ring_mask):
            ring_mask = None
            ring_outer_bbox = None

    image_out_dir = out_dir / image_id
    image_out_dir.mkdir(parents=True, exist_ok=True)

    toolkit.save_qc_overlay(
        image=original_crop,
        breast_mask=mask_crop,
        out_path=image_out_dir / f"{image_id}_qc_original.png",
        tissue_patch_bbox=patch_res.bbox,
        lesion_bbox=primary_lesion,
        ring_mask=ring_mask,
        title=f"{image_id} | original",
    )
    toolkit.save_image_png(original_crop, image_out_dir / f"{image_id}_original_crop.png")

    rows: list[dict] = []
    per_image_summary: dict = {
        "image_id": image_id,
        "image_path": str(image_path),
        "source_type": load_res.source_type,
        "photometric_interpretation": load_res.photometric_interpretation,
        "pixel_spacing_mm": pixel_spacing_mm,
        "num_lesions_full": len(lesion_bboxes_full),
        "num_lesions_crop": len(lesion_bboxes_crop),
        "crop_bbox_full_image": crop_res.bbox,
        "normalization": {
            "low": norm_params.low,
            "high": norm_params.high,
            "lower_percentile": norm_params.lower_percentile,
            "upper_percentile": norm_params.upper_percentile,
            "based_on_mask": norm_params.based_on_mask,
        },
        "tissue_patch_bbox": patch_res.bbox,
        "tissue_patch_score": patch_res.score,
        "tissue_patch_contour_margin_used_px": patch_res.contour_margin_used_px,
        "primary_lesion_bbox": primary_lesion,
        "ring_outer_bbox": ring_outer_bbox,
        "variants": [],
    }

    for spec_idx, spec in enumerate(plan):
        seed = int(base_seed + spec_idx)
        degraded = toolkit.apply_degradation(
            image=original_crop,
            degradation_type=spec.degradation_type,
            severity=spec.severity,
            pixel_spacing_mm=pixel_spacing_mm,
            baseline_noise_std=baseline_noise_std,
            seed=seed,
            mask=mask_crop,
        )

        metrics = toolkit.compute_fixed_roi_metrics(
            original_img=original_crop,
            degraded_img=degraded,
            breast_mask=mask_crop,
            tissue_patch_bbox=patch_res.bbox,
            lesion_bbox=primary_lesion,
            ring_mask=ring_mask,
        )

        variant_name = f"{spec.degradation_type}_s{spec.severity}"
        degraded_png_path = image_out_dir / f"{image_id}_{variant_name}.png"
        toolkit.save_image_png(degraded, degraded_png_path)

        qc_path = None
        if spec.degradation_type == "none" or save_degraded_qc:
            qc_path = image_out_dir / f"{image_id}_qc_{variant_name}.png"
            toolkit.save_qc_overlay(
                image=degraded,
                breast_mask=mask_crop,
                out_path=qc_path,
                tissue_patch_bbox=patch_res.bbox,
                lesion_bbox=primary_lesion,
                ring_mask=ring_mask,
                title=f"{image_id} | {variant_name}",
            )

        row = {
            "image_id": image_id,
            "image_path": str(image_path),
            "source_type": load_res.source_type,
            "photometric_interpretation": load_res.photometric_interpretation,
            "has_finding": int(primary_lesion is not None),
            "finding_count_crop": len(lesion_bboxes_crop),
            "degradation_type": spec.degradation_type,
            "severity": int(spec.severity),
            "seed": seed,
            "patch_size": int(patch_size),
            "buffer_px": int(buffer_px),
            "mask_area_px": int(mask_crop.sum()),
            "tissue_patch_valid": 1,
            "lesion_roi_valid": int(primary_lesion is not None),
            "cnr_ring_valid": int(ring_mask is not None and np.any(ring_mask)),
            "tissue_patch_score": float(patch_res.score),
            "tissue_patch_contour_margin_used_px": int(patch_res.contour_margin_used_px),
            "normalization_low": float(norm_params.low),
            "normalization_high": float(norm_params.high),
            "qc_path": str(qc_path) if qc_path is not None else None,
            "variant_png": str(degraded_png_path),
        }
        row.update(spec.params)
        row.update(metrics)
        rows.append(row)

        per_image_summary["variants"].append({
            "variant_name": variant_name,
            "degradation_type": spec.degradation_type,
            "severity": int(spec.severity),
            "params": dict(spec.params),
            "seed": seed,
            "metrics": metrics,
            "variant_png": str(degraded_png_path),
            "qc_path": str(qc_path) if qc_path is not None else None,
        })

    with (image_out_dir / f"{image_id}_summary.json").open("w", encoding="utf-8") as f:
        json.dump(per_image_summary, f, indent=2)

    return rows, per_image_summary


def main():
    args = parse_args()
    toolkit = load_toolkit(TOOLKIT_PATH)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.image is not None:
        image_paths = [Path(args.image)]
    else:
        image_paths = read_image_list(Path(args.image_list))

    if not image_paths:
        raise ValueError("No images found to process.")

    annotations_csv = Path(args.annotations) if args.annotations else None

    if args.degradation is not None:
        plan = build_single_degradation_spec(toolkit, args.degradation, args.deg_severity)
    else:
        plan = toolkit.build_default_degradation_plan()

    all_rows: list[dict] = []
    summaries: list[dict] = []

    for idx, image_path in enumerate(image_paths):
        image_id_override = args.image_id if (idx == 0 and len(image_paths) == 1) else None
        rows, summary = process_one_image(
            image_path=image_path,
            toolkit=toolkit,
            out_dir=out_dir,
            annotations_csv=annotations_csv,
            image_id_override=image_id_override,
            patch_size=args.patch_size,
            buffer_px=args.buffer_px,
            base_seed=args.seed + idx * 1000,
            save_degraded_qc=args.save_degraded_qc,
            plan=plan,
            dicom_force_invert=args.dicom_force_invert,
        )
        all_rows.extend(rows)
        summaries.append(summary)
        print(f"Processed {idx + 1}/{len(image_paths)}: {image_path}")

    save_metrics_csv(all_rows, out_dir / "metrics_batch.csv")
    with (out_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump({"images": summaries}, f, indent=2)

    print("Done.")
    print(f"Output dir: {out_dir}")
    print(f"Images processed: {len(image_paths)}")
    print(f"Variants per image: {len(plan)}")
    print(f"Metrics CSV: {out_dir / 'metrics_batch.csv'}")


if __name__ == "__main__":
    main()
