#!/usr/bin/env python
"""
run_50_sample_batch.py
======================

Samples 50 random DICOM mammograms from the VinDr-Mammo dataset and runs
the full degradation + fixed-ROI metrics pipeline on each.

Usage
-----
    python run_50_sample_batch.py

Or with custom options:

    python run_50_sample_batch.py --n-samples 100 --seed 42 --output-dir ./results_100

Directory structure assumed
---------------------------
    <VINDR_ROOT>/images/
        <study_folder_1>/
            <image_1>.dicom
            <image_2>.dicom
            <image_3>.dicom
            <image_4>.dicom
        <study_folder_2>/
            ...
        (5000 folders, 4 DICOMs each)

    <VINDR_ROOT>/finding_annotations.csv   (optional, for lesion bboxes)
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Paths — adjust these if your layout differs
# ---------------------------------------------------------------------------
VINDR_ROOT = Path(
    r"D:\Mammo\vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0"
)
IMAGES_DIR = VINDR_ROOT / "images"
ANNOTATIONS_CSV = VINDR_ROOT / "finding_annotations.csv"

HERE = Path(__file__).resolve().parent
TOOLKIT_PATH = HERE / "roi_metrics.py"
RUNNER_PATH = HERE / "test_runner.py"


# ---------------------------------------------------------------------------
# Dynamic imports
# ---------------------------------------------------------------------------

def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# DICOM discovery
# ---------------------------------------------------------------------------

def discover_dicoms(images_dir: Path) -> list[Path]:
    """Find all .dicom files under the images directory."""
    patterns = ["*.dicom", "*.dcm", "*.DICOM", "*.DCM"]
    all_files: list[Path] = []
    for pattern in patterns:
        all_files.extend(images_dir.rglob(pattern))
    # deduplicate (case-insensitive on Windows)
    seen: set[str] = set()
    unique: list[Path] = []
    for p in sorted(all_files):
        key = str(p).lower()
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def sample_dicoms(
    all_dicoms: list[Path],
    n: int,
    seed: int,
    strategy: str = "random",
) -> list[Path]:
    """Sample n DICOMs. Strategy: 'random' or 'one_per_study'."""
    rng = np.random.default_rng(seed)

    if strategy == "one_per_study":
        # Group by parent folder (= study), pick one per study, then sample
        by_study: dict[str, list[Path]] = {}
        for p in all_dicoms:
            study_key = str(p.parent)
            by_study.setdefault(study_key, []).append(p)
        # Pick one random image per study
        one_per_study = [rng.choice(files) for files in by_study.values()]
        if len(one_per_study) < n:
            print(f"Warning: only {len(one_per_study)} studies, requested {n}.")
            n = len(one_per_study)
        indices = rng.choice(len(one_per_study), size=n, replace=False)
        return [one_per_study[i] for i in indices]

    # Pure random across all DICOMs
    if len(all_dicoms) < n:
        print(f"Warning: only {len(all_dicoms)} DICOMs found, requested {n}.")
        n = len(all_dicoms)
    indices = rng.choice(len(all_dicoms), size=n, replace=False)
    return [all_dicoms[i] for i in indices]


def load_lesion_image_ids(annotations_csv: Path) -> set[str]:
    """Read finding_annotations.csv and return image_ids that have real lesion bboxes."""
    lesion_ids: set[str] = set()
    with annotations_csv.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            finding = row.get("finding_categories", "")
            xmin = row.get("xmin", "").strip()
            if "No Finding" in finding:
                continue
            if not xmin or xmin.lower() in ("", "nan", "none"):
                continue
            image_id = row.get("image_id", "").strip()
            if image_id:
                lesion_ids.add(image_id)
    return lesion_ids


def sample_stratified(
    all_dicoms: list[Path],
    annotations_csv: Path,
    n_with_lesion: int,
    n_without_lesion: int,
    seed: int,
) -> list[Path]:
    """Sample DICOMs stratified by lesion presence."""
    rng = np.random.default_rng(seed)

    lesion_ids = load_lesion_image_ids(annotations_csv)
    print(f"Annotation stats: {len(lesion_ids)} image_ids with lesion bboxes")

    with_lesion: list[Path] = []
    without_lesion: list[Path] = []
    for p in all_dicoms:
        if p.stem in lesion_ids:
            with_lesion.append(p)
        else:
            without_lesion.append(p)

    print(f"Matched DICOMs: {len(with_lesion)} with lesion, "
          f"{len(without_lesion)} without lesion")

    n_w = min(n_with_lesion, len(with_lesion))
    n_wo = min(n_without_lesion, len(without_lesion))
    if n_w < n_with_lesion:
        print(f"Warning: only {len(with_lesion)} DICOMs with lesion, requested {n_with_lesion}")
    if n_wo < n_without_lesion:
        print(f"Warning: only {len(without_lesion)} DICOMs without lesion, requested {n_without_lesion}")

    idx_w = rng.choice(len(with_lesion), size=n_w, replace=False)
    idx_wo = rng.choice(len(without_lesion), size=n_wo, replace=False)

    sampled = [with_lesion[i] for i in idx_w] + [without_lesion[i] for i in idx_wo]
    rng.shuffle(sampled)
    return sampled


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Run 50-sample batch experiment")
    p.add_argument("--images-dir", type=Path, default=IMAGES_DIR,
                   help="Root directory containing study folders with DICOMs")
    p.add_argument("--annotations", type=Path, default=ANNOTATIONS_CSV,
                   help="Path to finding_annotations.csv")
    p.add_argument("--output-dir", type=Path, default=HERE / "batch_50_results",
                   help="Output directory for metrics and QC images")
    p.add_argument("--n-samples", type=int, default=None,
                   help="Total sample count (ignored when --n-with-lesion is set)")
    p.add_argument("--n-with-lesion", type=int, default=40,
                   help="Number of images WITH lesion annotations (default: 30)")
    p.add_argument("--n-without-lesion", type=int, default=10,
                   help="Number of images WITHOUT lesion annotations (default: 20)")
    p.add_argument("--seed", type=int, default=2024,
                   help="Random seed for reproducible sampling")
    p.add_argument("--strategy", choices=["stratified", "random", "one_per_study"],
                   default="stratified",
                   help="Sampling strategy: 'stratified' (default) = N with + M without lesion; "
                        "'random' = any DICOM; 'one_per_study' = max one per study")
    p.add_argument("--patch-size", type=int, default=96)
    p.add_argument("--buffer-px", type=int, default=20)
    p.add_argument("--save-degraded-qc", action="store_true",
                   help="Save QC overlay for every degraded variant (large output)")
    p.add_argument("--dicom-force-invert", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    # --- Load toolkit and runner helpers ---
    toolkit = _load_module("roi_metrics", TOOLKIT_PATH)
    runner = _load_module("test_runner", RUNNER_PATH)

    # --- Discover and sample ---
    print(f"Scanning for DICOMs in: {args.images_dir}")
    all_dicoms = discover_dicoms(args.images_dir)
    print(f"Found {len(all_dicoms)} DICOM files.")

    if not all_dicoms:
        print("ERROR: No DICOM files found. Check --images-dir path.")
        sys.exit(1)

    sampled = []
    if args.strategy == "stratified":
        if not args.annotations.exists():
            print("ERROR: stratified sampling requires --annotations CSV.")
            sys.exit(1)
        sampled = sample_stratified(
            all_dicoms, args.annotations,
            n_with_lesion=args.n_with_lesion,
            n_without_lesion=args.n_without_lesion,
            seed=args.seed,
        )
        print(f"Sampled {len(sampled)} images "
              f"({args.n_with_lesion} with lesion + {args.n_without_lesion} without, "
              f"seed={args.seed}).")
    else:
        n = args.n_samples or (args.n_with_lesion + args.n_without_lesion)
        sampled = sample_dicoms(all_dicoms, n, args.seed, args.strategy)
        print(f"Sampled {len(sampled)} images (strategy={args.strategy}, seed={args.seed}).")

    # --- Prepare output ---
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save the sample list for reproducibility
    sample_list_path = out_dir / "sampled_images.txt"
    with sample_list_path.open("w", encoding="utf-8") as f:
        for p in sampled:
            f.write(f"{p}\n")
    print(f"Sample list saved: {sample_list_path}")

    # --- Annotations ---
    annotations_csv = args.annotations if args.annotations.exists() else None
    if annotations_csv:
        print(f"Using annotations: {annotations_csv}")
    else:
        print("No annotations CSV found — CNR/delta_mu will be NaN for all images.")

    # --- Build degradation plan ---
    plan = toolkit.build_default_degradation_plan()
    print(f"Degradation plan: {len(plan)} variants per image "
          f"(1 baseline + {len(plan) - 1} degradations)")
    print(f"Total metric computations: {len(sampled)} × {len(plan)} = {len(sampled) * len(plan)}")
    print()

    # --- Process ---
    all_rows: list[dict] = []
    summaries: list[dict] = []
    errors: list[dict] = []

    t0 = time.time()
    for idx, image_path in enumerate(sampled):
        t_img = time.time()
        print(f"[{idx + 1}/{len(sampled)}] {image_path.name} ... ", end="", flush=True)

        try:
            rows, summary = runner.process_one_image(
                image_path=image_path,
                toolkit=toolkit,
                out_dir=out_dir,
                annotations_csv=annotations_csv,
                image_id_override=None,
                patch_size=args.patch_size,
                buffer_px=args.buffer_px,
                base_seed=args.seed + idx * 1000,
                save_degraded_qc=args.save_degraded_qc,
                plan=plan,
                dicom_force_invert=args.dicom_force_invert,
            )
            all_rows.extend(rows)
            summaries.append(summary)
            dt = time.time() - t_img
            n_lesions = summary.get("num_lesions_crop", 0)
            print(f"OK ({dt:.1f}s, {n_lesions} lesion(s))")

        except Exception as e:
            dt = time.time() - t_img
            print(f"FAILED ({dt:.1f}s): {e}")
            errors.append({
                "image_path": str(image_path),
                "error": str(e),
                "index": idx,
            })

    total_time = time.time() - t0

    # --- Save batch results ---
    runner.save_metrics_csv(all_rows, out_dir / "metrics_batch.csv")

    with (out_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "n_samples": len(sampled),
                "n_with_lesion": args.n_with_lesion,
                "n_without_lesion": args.n_without_lesion,
                "seed": args.seed,
                "strategy": args.strategy,
                "patch_size": args.patch_size,
                "buffer_px": args.buffer_px,
                "variants_per_image": len(plan),
                "annotations": str(annotations_csv),
            },
            "images": summaries,
            "errors": errors,
        }, f, indent=2)

    if errors:
        with (out_dir / "errors.json").open("w", encoding="utf-8") as f:
            json.dump(errors, f, indent=2)

    # --- Summary ---
    print()
    print("=" * 60)
    print(f"  BATCH COMPLETE")
    print(f"  Images processed:  {len(summaries)}/{len(sampled)}")
    print(f"  Errors:            {len(errors)}")
    print(f"  Total variants:    {len(all_rows)}")
    print(f"  Total time:        {total_time:.0f}s ({total_time / max(1, len(sampled)):.1f}s/image)")
    print(f"  Output directory:  {out_dir}")
    print(f"  Metrics CSV:       {out_dir / 'metrics_batch.csv'}")
    print(f"  Run summary:       {out_dir / 'run_summary.json'}")
    print(f"  Sample list:       {sample_list_path}")
    if errors:
        print(f"  Error log:         {out_dir / 'errors.json'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
