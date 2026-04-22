#!/bin/bash
#SBATCH -J fairness_audit
#SBATCH -o status/fairness_audit_%A_%a.o
#SBATCH -e status/fairness_audit_%A_%a.e
#SBATCH -p gh
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -t 2:00:00
#SBATCH -A ASC25102
#SBATCH --array=0-11

module load cuda/12.4

set -e
cd $SCRATCH
eval "$(conda shell.bash hook)"
conda activate bb_audit_dpsgd
cd bb-audit-dpsgd

mkdir -p status

MODEL_NAME="cnn"
LR=0.1
BATCH_SIZE=4000
BLOCK_SIZE=4000
AUG_MULT=1
N_REPS=4
N_EPOCHS=100
SEED=0
TARGET_TYPE="blank"
DEFENSE_K=5
DEFENSE_SCORE_FN="grad_norm_unclipped"
DEFENSE_SCORE_NORM="linf"
PARENT_OUT="fairness_audits"

# 6 majority_pct values × 2 defense conditions = 12 tasks
# task_id = split_idx * 2 + defense_idx
MAJORITY_PCTS=(0.995 0.95 0.90 0.85 0.80 0.75)

SPLIT_IDX=$(( SLURM_ARRAY_TASK_ID / 2 ))
DEFENSE_IDX=$(( SLURM_ARRAY_TASK_ID % 2 ))

MAJORITY_PCT="${MAJORITY_PCTS[$SPLIT_IDX]}"

if [ "${DEFENSE_IDX}" -eq 0 ]; then
    DEFENSE="false"
    OUT_NAME="colored_mnist_maj${MAJORITY_PCT}_no_defense"
else
    DEFENSE="true"
    OUT_NAME="colored_mnist_maj${MAJORITY_PCT}_defense"
fi

echo "=========================================="
echo "Task ${SLURM_ARRAY_TASK_ID}: ${OUT_NAME}"
echo "Model: ${MODEL_NAME}  N_reps: ${N_REPS}"
echo "majority_pct: ${MAJORITY_PCT}  Defense: ${DEFENSE}"
echo "=========================================="

DEFENSE_ARGS=""
if [ "${DEFENSE}" = "true" ]; then
    DEFENSE_ARGS="--defense --defense_k ${DEFENSE_K} --defense_score_fn ${DEFENSE_SCORE_FN} --defense_score_norm ${DEFENSE_SCORE_NORM}"
fi

torchrun --nnodes=1 --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:$((29500 + SLURM_ARRAY_TASK_ID)) \
    fairness_audit.py \
    --data_name colored_mnist \
    --model_name "${MODEL_NAME}" \
    --n_reps "${N_REPS}" \
    --n_epochs "${N_EPOCHS}" \
    --lr "${LR}" \
    --batch_size "${BATCH_SIZE}" \
    --block_size "${BLOCK_SIZE}" \
    --aug_mult "${AUG_MULT}" \
    --epsilon -1 \
    --max_grad_norm -1 \
    --target_type "${TARGET_TYPE}" \
    --seed "${SEED}" \
    --fixed_init \
    --sampling poisson \
    --holdout_audit \
    --majority_pct "${MAJORITY_PCT}" \
    ${DEFENSE_ARGS} \
    --out "${PARENT_OUT}/${OUT_NAME}"

echo "Task ${SLURM_ARRAY_TASK_ID} completed: ${OUT_NAME}"
