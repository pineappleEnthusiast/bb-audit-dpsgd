#!/bin/bash
#SBATCH -J aug_consistency_mnist_mislabeled
#SBATCH -o status/aug_consistency_mnist_mislabeled_%A_%a.o
#SBATCH -e status/aug_consistency_mnist_mislabeled_%A_%a.e
#SBATCH -p gh
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -t 8:00:00
#SBATCH -A ASC25102
#SBATCH --array=0-1

module load cuda/12.4

set -e
cd $SCRATCH
eval "$(conda shell.bash hook)"
conda activate bb_audit_dpsgd
cd bb-audit-dpsgd

mkdir -p status

# Fixed hyperparameters
DATA_NAME="mnist"
MODEL_NAME="cnn"
LR=3
BATCH_SIZE=4000
BLOCK_SIZE=4000
N_REPS=400
N_EPOCHS=100
MAX_GRAD_NORM=1
DELTA=1e-5
SEED=0
TARGET_TYPE="mislabeled"
AUG_K=20
DEFENSE_K=5
DEFENSE_SCORE_FN="grad_norm_unclipped"
DEFENSE_SCORE_NORM="linf"
PARENT_OUT="aug_consistency_results"

# Array: 10 tasks — (eps=10) x (no_defense, defense)
# Format: "epsilon|defense|output_name"
experiments=(
    "10.0|false|mnist_mislabeled_no_defense_eps10"
    "10.0|true|mnist_mislabeled_defense_eps10"
)

IFS='|' read -r EPSILON DEFENSE OUT_NAME <<< "${experiments[$SLURM_ARRAY_TASK_ID]}"

echo "=========================================="
echo "Task ${SLURM_ARRAY_TASK_ID}: ${OUT_NAME}"
echo "Dataset: ${DATA_NAME}  Model: ${MODEL_NAME}"
echo "Epsilon: ${EPSILON}  Defense: ${DEFENSE}"
echo "Aug-consistency k: ${AUG_K}"
echo "=========================================="

DEFENSE_ARGS=""
if [ "${DEFENSE}" = "true" ]; then
    DEFENSE_ARGS="--defense --defense_k ${DEFENSE_K} --defense_score_fn ${DEFENSE_SCORE_FN} --defense_score_norm ${DEFENSE_SCORE_NORM}"
fi

python aug_consistency_audit.py \
    --data_name ${DATA_NAME} \
    --model_name ${MODEL_NAME} \
    --n_reps ${N_REPS} \
    --n_epochs ${N_EPOCHS} \
    --lr ${LR} \
    --batch_size ${BATCH_SIZE} \
    --block_size ${BLOCK_SIZE} \
    --epsilon ${EPSILON} \
    --delta ${DELTA} \
    --max_grad_norm ${MAX_GRAD_NORM} \
    --target_type ${TARGET_TYPE} \
    --seed ${SEED} \
    --fixed_init \
    --sampling poisson \
    --holdout_audit \
    --aug_k ${AUG_K} \
    ${DEFENSE_ARGS} \
    --out ${PARENT_OUT}/${OUT_NAME}

echo "Task ${SLURM_ARRAY_TASK_ID} completed: ${OUT_NAME}"

sacct -u $USER -o JobID,JobName,Partition,AllocCPUS,State,Elapsed,Timelimit,Start,End
