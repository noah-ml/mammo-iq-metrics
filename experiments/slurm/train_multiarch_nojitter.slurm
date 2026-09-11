#!/bin/bash
#SBATCH --job-name=multiarch_nojitter
#SBATCH --account=lorch01
#SBATCH --partition=equipment_typeG
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-7
#SBATCH --output=/scratch/lorch01/output/multiarch/%A_%a/train_%A_%a.out
#SBATCH --error=/scratch/lorch01/output/multiarch/%A_%a/train_%A_%a.err

# ---------------------------------------------------------------------------
# Paths. The defaults below reproduce the runs as submitted on the PTB cluster.
# Export these before submitting to run elsewhere. The #SBATCH --output and
# --error paths above are parsed by SLURM before the shell starts, so they
# cannot use variables and must be edited directly.
# ---------------------------------------------------------------------------
: "${MAMMO_SCRATCH:=/scratch/lorch01}"
: "${MAMMO_SCRIPTS:=/home/lorch01/scripts}"
#
# Multi-architecture x multi-seed robustness campaign (NO ColorJitter).
# 8 NEW runs; ConvNeXt seed 42 already exists (run 956694) and is NOT re-trained here.
# 3x3 fully-crossed design (Adam A1 + Supervisor P1), seeds 42/43/44:
#   idx 0: convnext_tiny  seed 43
#   idx 1: convnext_tiny  seed 44
#   idx 2: resnet18       seed 42
#   idx 3: resnet18       seed 43
#   idx 4: resnet18       seed 44
#   idx 5: efficientnet_b4 seed 42
#   idx 6: efficientnet_b4 seed 43
#   idx 7: efficientnet_b4 seed 44
# Everything else matches the primary protocol (train_convnext_nojitter.slurm) verbatim.

# ── task -> (arch, seed) map ─────────────────────────────────────────────────
ARCHS=(convnext_tiny convnext_tiny resnet18 resnet18 resnet18 efficientnet_b4 efficientnet_b4 efficientnet_b4)
SEEDS=(43            44            42       43       44       42              43              44)

ARCH=${ARCHS[$SLURM_ARRAY_TASK_ID]}
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

# Per-arch batch size — matches the thesis Table tab:training-config exactly
# (ResNet-18 = 96; EfficientNet-B4 and ConvNeXt-Tiny = 48). These are the same
# batches used for the already-trained with-jitter baselines, so the no-jitter
# runs stay protocol-consistent.
case "$ARCH" in
  convnext_tiny)   BATCH=48 ;;
  resnet18)        BATCH=96 ;;
  efficientnet_b4) BATCH=48 ;;
esac

SCRIPT_DIR="$MAMMO_SCRIPTS"
DATA_DIR="$MAMMO_SCRATCH/vindr_uploads/vindr_tar_shards_1024x384_45positive"
OUT_DIR="$MAMMO_SCRATCH/output/multiarch/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

mkdir -p "$OUT_DIR"

source ~/miniforge/bin/activate mammo_torch
export WANDB_MODE=offline

echo "======================================================="
echo " Multi-arch no-jitter — array ${SLURM_ARRAY_JOB_ID} task ${SLURM_ARRAY_TASK_ID}"
echo " arch=${ARCH}  seed=${SEED}  batch=${BATCH}"
echo " Node: $(hostname)   GPUs: $CUDA_VISIBLE_DEVICES"
echo " Start: $(date)"
echo "======================================================="

python "$SCRIPT_DIR/mammo-18-v3.py" \
    --data-dir       "$DATA_DIR"          \
    --output-dir     "$OUT_DIR"           \
    --arch           "$ARCH"              \
    --batch-size     "$BATCH"             \
    --epochs         50                   \
    --freeze-epochs  5                    \
    --lr-backbone    3e-4                 \
    --lr-head        1e-3                 \
    --weight-decay   1e-3                 \
    --dropout        0.3                  \
    --label-smoothing 0.1                 \
    --oversample-rate 0.20                \
    --warmup-epochs  3                    \
    --patience       15                   \
    --rolling-window 5                    \
    --num-workers    8                    \
    --seed           "$SEED"              \
    --no-colorjitter                      \
    --no-tta                              \
    --no-compile                          \
    --wandb-project  mammo-robustness     \
    --wandb-entity   noah-ml-tu-berlin    \
    --wandb-name     "${ARCH}-1024x384-45pos-nojitter-s${SEED}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" \
    --wandb-offline

echo ""
echo "======================================================="
echo " Done: $(date)  (arch=${ARCH} seed=${SEED})"
echo "======================================================="
