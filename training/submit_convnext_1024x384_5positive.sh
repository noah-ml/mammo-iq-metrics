#!/bin/bash
#SBATCH --job-name=convnext_5pos
#SBATCH --partition=YOUR_PARTITION        # <-- fill in
#SBATCH --account=YOUR_ACCOUNT            # <-- fill in (remove if not needed)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --output=/scratch/lorch01/output/convnext_1024x384_5positive/%j/slurm.%j.out
#SBATCH --error=/scratch/lorch01/output/convnext_1024x384_5positive/%j/slurm.%j.err

# ── Preprocessing (run locally before submitting) ─────────────────────────────
#
# Step 1 — generate strict-label split CSV (BI-RADS 5 only = positive):
#
#   python make_kheiron_splits.py \
#       --input-csv  master_splits_1024x832.csv \
#       --output-csv master_splits_1024x384_5positive.csv \
#       --new-data-dir D:/vindr_tar_shards_1024x384_5positive \
#       --birads-csv "D:/Mammo/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0/breast-level_annotations.csv"
#
# This copies train/val/test assignments verbatim from master_splits_1024x832.csv
# (same patient-level split as the 45positive ConvNeXt run) and relabels
# BI-RADS 4 → 0, BI-RADS 5 → 1.
#
# Step 2 — create 1024×384 TAR shards with strict labels:
#
#   python dicom_to_tar_shards.py \
#       --existing-splits master_splits_1024x384_5positive.csv \
#       --output-dir D:/vindr_tar_shards_1024x384_5positive \
#       --target-height 1024 \
#       --target-width  384  \
#       --samples-per-shard 500 \
#       --num-workers 8 \
#       --verify
#
# Step 3 — upload to cluster scratch:
#
#   rsync -av --progress \
#       D:/vindr_tar_shards_1024x384_5positive/ \
#       lorch01@<cluster>:/scratch/lorch01/vindr_uploads/vindr_tar_shards_1024x384_5positive/
#
# ─────────────────────────────────────────────────────────────────────────────

# ── Environment ───────────────────────────────────────────────────────────────
source ~/anaconda3/etc/profile.d/conda.sh
conda activate YOUR_ENV                   # <-- fill in

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR=/scratch/lorch01/vindr_uploads/vindr_tar_shards_1024x384_5positive
OUTPUT_DIR=/scratch/lorch01/output/convnext_1024x384_5positive/${SLURM_JOB_ID}
SCRIPT_DIR=$(dirname "$0")

mkdir -p "$OUTPUT_DIR"

# ── Run ───────────────────────────────────────────────────────────────────────
python "$SCRIPT_DIR/mammo-18-v3.py" \
    --data-dir          "$DATA_DIR"                                              \
    --output-dir        "$OUTPUT_DIR"                                            \
    --arch              convnext_tiny                                            \
    --batch-size        48                                                       \
    --epochs            50                                                       \
    --dropout           0.3                                                      \
    --label-smoothing   0.1                                                      \
    --oversample-rate   0.20                                                     \
    --weight-decay      1e-3                                                     \
    --rolling-window    5                                                        \
    --patience          15                                                       \
    --no-tta                                                                     \
    --wandb-project     resnet18-1024                                            \
    --wandb-entity      noah-ml-tu-berlin                                        \
    --wandb-name        "convnext_tiny_1024x384_birads5only_fixedtest_${SLURM_JOB_ID}" \
    --wandb-offline

# ── Post-run sanity checks ────────────────────────────────────────────────────
echo ""
echo "=== POST-RUN SANITY CHECKS ==="

python3 <<EOF
import pandas as pd, sys

manifest_path = "${DATA_DIR}/metadata/shard_manifest.csv"
pred_path     = "${OUTPUT_DIR}/test_predictions.csv"

print("\n--- Split / label counts ---")
try:
    df = pd.read_csv(manifest_path)
    for split in ["train", "val", "test"]:
        sub = df[df["split"] == split]
        n   = len(sub)
        pos = int(sub["label"].sum())
        neg = n - pos
        pct = 100.0 * pos / n if n > 0 else float("nan")
        print(f"  {split:<5}  n={n:5d}  pos={pos:4d}  neg={neg:5d}  pos%={pct:.1f}%")
    total = len(df)
    total_pos = int(df["label"].sum())
    print(f"  {'ALL':<5}  n={total:5d}  pos={total_pos:4d}  neg={total-total_pos:5d}  pos%={100*total_pos/total:.1f}%")
except Exception as e:
    print(f"  WARNING: cannot read manifest: {e}", file=sys.stderr)

print("\n--- test_predictions.csv row count ---")
try:
    df_pred = pd.read_csv(pred_path)
    df_man  = pd.read_csv(manifest_path)
    expected = len(df_man[df_man["split"] == "test"])
    status = "OK" if len(df_pred) == expected else "MISMATCH"
    print(f"  rows={len(df_pred)}  expected={expected}  → {status}")
except Exception as e:
    print(f"  WARNING: cannot read test_predictions.csv: {e}", file=sys.stderr)
EOF

echo ""
echo "To sync W&B offline run to cloud:"
echo "  wandb sync ${OUTPUT_DIR}/wandb/latest-run"
