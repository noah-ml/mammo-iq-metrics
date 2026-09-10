#!/bin/bash
#SBATCH --job-name=mammo_deg_eval
#SBATCH --account=lorch01
#SBATCH --partition=equipment_typeG
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-8
#SBATCH --output=/scratch/lorch01/output/multiarch_eval/eval_%A_%a.out
#SBATCH --error=/scratch/lorch01/output/multiarch_eval/eval_%A_%a.err
#
# Degradation evaluation for the nine released checkpoints
# (3 architectures x 3 seeds). Each task writes one per-image results CSV:
# 4,000 test images x 31 variants = 124,000 rows.
#
# Each row carries the model logit and probability alongside the image quality
# metrics computed on the same variant, which is what makes the per-image
# paired analysis of Research Question 2 possible.
#
# ---------------------------------------------------------------------------
# Paths. The defaults below reproduce the runs as submitted on the PTB cluster.
# Export these before submitting to run elsewhere. The #SBATCH --output and
# --error paths above are parsed by SLURM before the shell starts, so they
# cannot use variables and must be edited directly.
# ---------------------------------------------------------------------------
: "${MAMMO_SCRATCH:=/scratch/lorch01}"
: "${MAMMO_SCRIPTS:=/home/lorch01/scripts}"
: "${MAMMO_CKPT_DIR:=$MAMMO_SCRATCH/checkpoints}"

SHARDS_DIR="$MAMMO_SCRATCH/vindr_uploads/vindr_tar_shards_1024x384_45positive"
SPLITS_CSV="$MAMMO_SCRIPTS/master_splits_1024x384_complete.csv"
LESION_ROI_CSV="$MAMMO_SCRIPTS/lesion_roi_lookup_1024x384.csv"
OUT_DIR="$MAMMO_SCRATCH/output/multiarch_eval"

# arch:seed:checkpoint filename. The checkpoint chosen for each run is the one
# with the highest validation AUC, which is the selection rule used throughout;
# the validation AUC is part of the filename.
RUNS=(
  "convnext_tiny:42:convnext_tiny_seed42__convnext_nojitter_ckpt_epoch006_auc0.8103.pth"
  "convnext_tiny:43:convnext_tiny_seed43__ckpt_epoch006_auc0.8161.pth"
  "convnext_tiny:44:convnext_tiny_seed44__ckpt_epoch007_auc0.8292.pth"
  "resnet18:42:resnet18_seed42__ckpt_epoch022_auc0.7952.pth"
  "resnet18:43:resnet18_seed43__ckpt_epoch016_auc0.8023.pth"
  "resnet18:44:resnet18_seed44__ckpt_epoch015_auc0.7751.pth"
  "efficientnet_b4:42:efficientnet_b4_seed42__ckpt_epoch006_auc0.7677.pth"
  "efficientnet_b4:43:efficientnet_b4_seed43__ckpt_epoch008_auc0.8147.pth"
  "efficientnet_b4:44:efficientnet_b4_seed44__ckpt_epoch008_auc0.7717.pth"
)

IFS=':' read -r ARCH SEED CKPT_FILE <<< "${RUNS[$SLURM_ARRAY_TASK_ID]}"

mkdir -p "$OUT_DIR"

source ~/miniforge/bin/activate mammo_torch
export WANDB_MODE=offline

echo "======================================================="
echo " Degradation evaluation - task ${SLURM_ARRAY_TASK_ID}"
echo " arch=${ARCH}  seed=${SEED}"
echo " ckpt=${CKPT_FILE}"
echo " Node: $(hostname)   GPUs: $CUDA_VISIBLE_DEVICES"
echo " Start: $(date)"
echo "======================================================="

python "$MAMMO_SCRIPTS/evaluate_degradations.py" \
    --checkpoint     "$MAMMO_CKPT_DIR/$CKPT_FILE"                     \
    --shards-dir     "$SHARDS_DIR"                                    \
    --splits-csv     "$SPLITS_CSV"                                    \
    --lesion-roi-csv "$LESION_ROI_CSV"                                \
    --arch           "$ARCH"                                          \
    --output         "$OUT_DIR/${ARCH}_s${SEED}_degradation_results.csv"

status=$?
echo "Finished: $(date)  exit=${status}"
exit $status
