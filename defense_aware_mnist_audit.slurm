#!/bin/bash
#SBATCH -J defense_aware_mnist_audit
#SBATCH -o status/defense_aware_mnist_audit_%A_%a.o
#SBATCH -e status/defense_aware_mnist_audit_%A_%a.e
#SBATCH -p gh
#SBATCH -N 5
#SBATCH --ntasks-per-node=1
#SBATCH -t 5:00:00
#SBATCH -A ASC25081
#SBATCH --array=0-3

module load cuda/12.4

set -e
cd $SCRATCH
eval "$(conda shell.bash hook)"
conda activate bb_audit_dpsgd
cd bb-audit-dpsgd

mkdir -p status

DATA_NAME="mnist"
MODEL_NAME="cnn"
LR=3
BATCH_SIZE=4000
BLOCK_SIZE=4000
N_REPS=400
N_EPOCHS=100
MAX_GRAD_NORM=1
EPSILON=10.0
DELTA=1e-5
SEED=0
DEFENSE_K=5
DEFENSE_SCORE_FN="grad_norm_unclipped"
DEFENSE_SCORE_NORM="linf"
CANARY_PT="defense_aware_canaries/mnist/canary.pt"
PARENT_OUT="defense_aware_mnist_audit"

mkdir -p ${PARENT_OUT}

# Format: "canary_type|defense|output_name"
#   0: naive mislabeled, no defense
#   1: naive mislabeled, with defense
#   2: defense-aware winner, no defense
#   3: defense-aware winner, with defense
experiments=(
    "mislabeled|false|mnist_mislabeled_no_defense"
    "mislabeled|true|mnist_mislabeled_defense"
    "pt|false|mnist_defense_aware_no_defense"
    "pt|true|mnist_defense_aware_defense"
)

IFS='|' read -r CANARY_TYPE DEFENSE OUT_NAME <<< "${experiments[$SLURM_ARRAY_TASK_ID]}"

if [ "${CANARY_TYPE}" = "pt" ]; then
    CANARY_ARGS="--canary_pt ${CANARY_PT} --target_type blank"
else
    CANARY_ARGS="--target_type mislabeled"
fi

if [ "${DEFENSE}" = "true" ]; then
    DEFENSE_ARGS="--defense --defense_k ${DEFENSE_K} --defense_score_fn ${DEFENSE_SCORE_FN} --defense_score_norm ${DEFENSE_SCORE_NORM}"
else
    DEFENSE_ARGS=""
fi

echo "=========================================="
echo "Job Array Task ID : ${SLURM_ARRAY_TASK_ID}"
echo "Canary type       : ${CANARY_TYPE}"
echo "Defense enabled   : ${DEFENSE}"
echo "Output            : ${OUT_NAME}"
echo "=========================================="

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
    --epsilon '"${EPSILON}"' \
    --delta '"${DELTA}"' \
    --max_grad_norm '"${MAX_GRAD_NORM}"' \
    --seed '"${SEED}"' \
    --fixed_init \
    --sampling poisson \
    --holdout_audit \
    '"${CANARY_ARGS}"' \
    '"${DEFENSE_ARGS}"' \
    --out '"${PARENT_OUT}/${OUT_NAME}"''

echo "Task ${SLURM_ARRAY_TASK_ID} completed: ${OUT_NAME}"
