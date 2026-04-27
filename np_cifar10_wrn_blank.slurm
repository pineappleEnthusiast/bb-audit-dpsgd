#!/bin/bash
#SBATCH -J np_cifar10_wrn_blank
#SBATCH -o status/np_cifar10_wrn_blank_%A_%a.o
#SBATCH -e status/np_cifar10_wrn_blank_%A_%a.e
#SBATCH -p gh
#SBATCH -N 5
#SBATCH --ntasks-per-node=1
#SBATCH -t 30:00:00
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
MODEL_NAME="wideresnet"
LR=0.1
BATCH_SIZE=3125
BLOCK_SIZE=1000
AUG_MULT=1
N_REPS=400
N_EPOCHS=100
SEED=0
TARGET_TYPE="blank"
DEFENSE_K=5
DEFENSE_SCORE_FN="grad_norm_unclipped"
DEFENSE_SCORE_NORM="linf"
PARENT_OUT="non_private_audits"

experiments=(
    "false|cifar10_wrn_no_defense_non_private"
    "true|cifar10_wrn_defense_non_private"
)

IFS='|' read -r DEFENSE OUT_NAME <<< "${experiments[$SLURM_ARRAY_TASK_ID]}"

echo "=========================================="
echo "Task ${SLURM_ARRAY_TASK_ID}: ${OUT_NAME}"
echo "Dataset: ${DATA_NAME}  Model: ${MODEL_NAME}"
echo "Non-private  Defense: ${DEFENSE}"
echo "=========================================="

DEFENSE_ARGS=""
if [ "${DEFENSE}" = "true" ]; then
    DEFENSE_ARGS="--defense --defense_k ${DEFENSE_K} --defense_score_fn ${DEFENSE_SCORE_FN} --defense_score_norm ${DEFENSE_SCORE_NORM}"
fi

srun --ntasks=$SLURM_NTASKS --nodes=$SLURM_JOB_NUM_NODES bash -c '
  MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1);
  MASTER_PORT=$((29500 + '"${SLURM_ARRAY_TASK_ID}"'));
  torchrun --nnodes=$SLURM_JOB_NUM_NODES --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    parallel_audit_model.py \
    --data_name '"${DATA_NAME}"' \
    --model_name '"${MODEL_NAME}"' \
    --n_reps '"${N_REPS}"' \
    --n_epochs '"${N_EPOCHS}"' \
    --lr '"${LR}"' \
    --batch_size '"${BATCH_SIZE}"' \
    --block_size '"${BLOCK_SIZE}"' \
    --aug_mult '"${AUG_MULT}"' \
    --epsilon -1 \
    --max_grad_norm -1 \
    --target_type '"${TARGET_TYPE}"' \
    --seed '"${SEED}"' \
    --fixed_init \
    --sampling poisson \
    --holdout_audit \
    '"${DEFENSE_ARGS}"' \
    --out '"${PARENT_OUT}"'/'"${OUT_NAME}"''

echo "Task ${SLURM_ARRAY_TASK_ID} completed: ${OUT_NAME}"
