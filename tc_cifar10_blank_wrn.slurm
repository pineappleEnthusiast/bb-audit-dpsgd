#!/bin/bash
#SBATCH -J tc_cifar10_blank_wrn
#SBATCH -o status/tc_cifar10_blank_wrn_%A_%a.o
#SBATCH -e status/tc_cifar10_blank_wrn_%A_%a.e
#SBATCH -p gh
#SBATCH -N 20
#SBATCH --ntasks-per-node=1
#SBATCH -t 30:00:00
#SBATCH -A ASC25102
#SBATCH --array=0-9

module load cuda/12.4

set -e
cd $SCRATCH
eval "$(conda shell.bash hook)"
conda activate bb_audit_dpsgd
cd bb-audit-dpsgd

mkdir -p status

# Fixed hyperparameters
DATA_NAME="cifar10"
MODEL_NAME="wideresnet"
LR=3
BATCH_SIZE=3125
BLOCK_SIZE=1000
AUG_MULT=1
N_REPS=400
N_EPOCHS=100
MAX_GRAD_NORM=1
DELTA=1e-5
SEED=0
TARGET_TYPE="blank"
DEFENSE_K=5
DEFENSE_SCORE_FN="grad_norm_unclipped"
DEFENSE_SCORE_NORM="linf"
PARENT_OUT="tradeoff_curves"

# Array: 10 tasks — (eps=2,4,6,8,10) x (no_defense, defense)
# Format: "epsilon|defense|output_name"
experiments=(
    "2.0|false|cifar10_wrn_no_defense_eps2"
    "2.0|true|cifar10_wrn_defense_eps2"
    "4.0|false|cifar10_wrn_no_defense_eps4"
    "4.0|true|cifar10_wrn_defense_eps4"
    "6.0|false|cifar10_wrn_no_defense_eps6"
    "6.0|true|cifar10_wrn_defense_eps6"
    "8.0|false|cifar10_wrn_no_defense_eps8"
    "8.0|true|cifar10_wrn_defense_eps8"
    "10.0|false|cifar10_wrn_no_defense_eps10"
    "10.0|true|cifar10_wrn_defense_eps10"
)

IFS='|' read -r EPSILON DEFENSE OUT_NAME <<< "${experiments[$SLURM_ARRAY_TASK_ID]}"

echo "=========================================="
echo "Task ${SLURM_ARRAY_TASK_ID}: ${OUT_NAME}"
echo "Dataset: ${DATA_NAME}  Model: ${MODEL_NAME}"
echo "Epsilon: ${EPSILON}  Defense: ${DEFENSE}"
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
    --epsilon '"${EPSILON}"' \
    --delta '"${DELTA}"' \
    --max_grad_norm '"${MAX_GRAD_NORM}"' \
    --target_type '"${TARGET_TYPE}"' \
    --seed '"${SEED}"' \
    --fixed_init \
    --sampling poisson \
    --holdout_audit \
    '"${DEFENSE_ARGS}"' \
    --out '"${PARENT_OUT}"'/'"${OUT_NAME}"''

echo "Task ${SLURM_ARRAY_TASK_ID} completed: ${OUT_NAME}"
