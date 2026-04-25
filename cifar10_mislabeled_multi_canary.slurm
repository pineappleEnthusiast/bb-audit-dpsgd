#!/bin/bash
#SBATCH -J cifar10_mislabeled_multi
#SBATCH -o status/cifar10_mislabeled_multi_%A_%a.o
#SBATCH -e status/cifar10_mislabeled_multi_%A_%a.e
#SBATCH -p gh
#SBATCH -N 10
#SBATCH --ntasks-per-node=1
#SBATCH -t 10:00:00
#SBATCH -A ASC25102
#SBATCH --array=0-1

module load cuda/12.4

set -e
cd $SCRATCH
eval "$(conda shell.bash hook)"
conda activate bb_audit_dpsgd
cd bb-audit-dpsgd

mkdir -p status

DATA_NAME="cifar10"
MODEL_NAME="cnn"
N_REPS=400
N_EPOCHS=100
LR=3
MAX_GRAD_NORM=1.0
EPSILON=10.0
DELTA=1e-5
TARGET_TYPE="mislabeled"
BATCH_SIZE=3300
BLOCK_SIZE=3300
N_CANARIES=500
AUG_MULT=1
SEED=0
DEFENSE_K=5
DEFENSE_SCORE_FN="prediction_margin"
DEFENSE_SCORE_NORM="linf"

experiments=(
    "false|cifar10_mislabeled_multi_no_defense_eps10"
    "true|cifar10_mislabeled_multi_defense_eps10"
)

IFS='|' read -r DEFENSE OUT_NAME <<< "${experiments[$SLURM_ARRAY_TASK_ID]}"

DEFENSE_ARGS=""
if [ "${DEFENSE}" = "true" ]; then
    DEFENSE_ARGS="--defense --defense_k ${DEFENSE_K} --defense_score_fn ${DEFENSE_SCORE_FN} --defense_score_norm ${DEFENSE_SCORE_NORM}"
fi

echo "=========================================="
echo "Task ${SLURM_ARRAY_TASK_ID}: DEFENSE=${DEFENSE}"
echo "Dataset: ${DATA_NAME}  Model: ${MODEL_NAME}  ε=${EPSILON}"
echo "N_CANARIES: ${N_CANARIES}  Target: ${TARGET_TYPE}"
echo "Output: ${OUT_NAME}"
echo "=========================================="

srun --ntasks=$SLURM_NTASKS --nodes=$SLURM_JOB_NUM_NODES bash -c '
  MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1);
  MASTER_PORT=$((29500 + '"${SLURM_ARRAY_TASK_ID}"'));
  torchrun --nnodes=$SLURM_JOB_NUM_NODES --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    parallel_audit_multi_canary.py \
    --data_name '"${DATA_NAME}"' \
    --model_name '"${MODEL_NAME}"' \
    --n_reps '"${N_REPS}"' \
    --n_epochs '"${N_EPOCHS}"' \
    --lr '"${LR}"' \
    --max_grad_norm '"${MAX_GRAD_NORM}"' \
    --epsilon '"${EPSILON}"' \
    --delta '"${DELTA}"' \
    --target_type '"${TARGET_TYPE}"' \
    --batch_size '"${BATCH_SIZE}"' \
    --block_size '"${BLOCK_SIZE}"' \
    --n_canaries '"${N_CANARIES}"' \
    --aug_mult '"${AUG_MULT}"' \
    --sampling poisson \
    --holdout_audit \
    --seed '"${SEED}"' \
    --fixed_init \
    '"${DEFENSE_ARGS}"' \
    --out '"${OUT_NAME}"''

echo "Task ${SLURM_ARRAY_TASK_ID} completed: ${OUT_NAME}"
