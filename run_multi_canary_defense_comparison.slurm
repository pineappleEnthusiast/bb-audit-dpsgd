#!/bin/bash
#SBATCH -J multi_canary_defense_comp
#SBATCH -o status/multi_canary_defense_comp_%A_%a.o
#SBATCH -e status/multi_canary_defense_comp_%A_%a.e
#SBATCH -p gh
#SBATCH -N 10
#SBATCH --ntasks-per-node=1
#SBATCH -t 20:00:00
#SBATCH -A ASC25102
#SBATCH --array=0-1

module load cuda/12.4

set -e
cd $SCRATCH
eval "$(conda shell.bash hook)"
conda activate bb_audit_dpsgd
cd bb-audit-dpsgd

# Common hyperparameters
DATA_NAME="cifar10"
MODEL_NAME="cnn"
N_REPS=400
N_EPOCHS=100
LR=3
MAX_GRAD_NORM=1.0
EPSILON=10.0
DELTA=1e-5
TARGET_TYPE="mislabeled"
BATCH_SIZE=3125
BLOCK_SIZE=3125
N_CANARIES=500
AUG_MULT=1
DEFENSE_K=5
DEFENSE_SCORE_FN="prediction_margin"

# Define experiment configurations
# Array format: "defense|output_name"
experiments=(
    "false|cifar10_multi_canary_no_defense"
    "true|cifar10_multi_canary_defense"
)

# Get current experiment configuration
IFS='|' read -r DEFENSE OUT_NAME <<< "${experiments[$SLURM_ARRAY_TASK_ID]}"

echo "=========================================="
echo "Job Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Dataset: ${DATA_NAME}"
echo "Model: ${MODEL_NAME}"
echo "N Canaries: ${N_CANARIES}"
echo "Defense: ${DEFENSE}"
echo "Defense Score Function: ${DEFENSE_SCORE_FN}"
echo "Output: ${OUT_NAME}"
echo "=========================================="

# Build defense arguments
DEFENSE_ARGS=""
if [ "${DEFENSE}" = "true" ]; then
    DEFENSE_ARGS="--defense --defense_k ${DEFENSE_K} --defense_score_fn ${DEFENSE_SCORE_FN} --defense_apply_ascent"
fi

# Run the audit
srun --ntasks=$SLURM_NTASKS --nodes=$SLURM_JOB_NUM_NODES bash -c '
  MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1);
  MASTER_PORT=$((29500 + '"${SLURM_ARRAY_TASK_ID}"'));
  RANK=$SLURM_PROCID;
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
    '"${DEFENSE_ARGS}"' \
    --out '"${OUT_NAME}"''

echo "=========================================="
echo "Task ${SLURM_ARRAY_TASK_ID} completed: ${OUT_NAME}"
echo "Results saved in: ${OUT_NAME}/"
echo "=========================================="
