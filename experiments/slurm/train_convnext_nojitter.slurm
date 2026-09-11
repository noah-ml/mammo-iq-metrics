#!/bin/bash
#SBATCH --job-name=convnext_nojitter
#SBATCH --account=lorch01
#SBATCH --partition=equipment_typeG
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/lorch01/output/convnext_nojitter/%j/train_%j.out

# ---------------------------------------------------------------------------
# Paths. The defaults below reproduce the runs as submitted on the PTB cluster.
# Export these before submitting to run elsewhere. The #SBATCH --output and
# --error paths above are parsed by SLURM before the shell starts, so they
# cannot use variables and must be edited directly.
# ---------------------------------------------------------------------------
: "${MAMMO_SCRATCH:=/scratch/lorch01}"
: "${MAMMO_SCRIPTS:=/home/lorch01/scripts}"
#SBATCH --error=/scratch/lorch01/output/convnext_nojitter/%j/train_%j.err

SCRIPT_DIR="$MAMMO_SCRIPTS"
DATA_DIR="$MAMMO_SCRATCH/vindr_uploads/vindr_tar_shards_1024x384_45positive"
OUT_DIR="$MAMMO_SCRATCH/output/convnext_nojitter/${SLURM_JOB_ID}"

mkdir -p "$OUT_DIR"

source ~/miniforge/bin/activate mammo_torch
export WANDB_MODE=offline

echo "======================================================="
echo " ConvNeXt-Tiny 45pos — no ColorJitter — job ${SLURM_JOB_ID}"
echo " Node: $(hostname)   GPUs: $CUDA_VISIBLE_DEVICES"
echo " Start: $(date)"
echo "======================================================="

python "$SCRIPT_DIR/mammo-18-v3.py" \
    --data-dir       "$DATA_DIR"          \
    --output-dir     "$OUT_DIR"           \
    --arch           convnext_tiny        \
    --batch-size     48                   \
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
    --seed           42                   \
    --no-colorjitter                      \
    --no-tta                              \
    --wandb-project  mammo-robustness     \
    --wandb-entity   noah-ml-tu-berlin    \
    --wandb-name     "convnext-tiny-1024x384-45pos-nojitter-${SLURM_JOB_ID}" \
    --wandb-offline

echo ""
echo "======================================================="
echo " Done: $(date)"
echo "======================================================="
