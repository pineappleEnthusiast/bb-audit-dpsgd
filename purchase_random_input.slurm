#!/bin/bash
#SBATCH -J purchase_random_input
#SBATCH -o status/purchase_random_input_%A_%a.o
#SBATCH -e status/purchase_random_input_%A_%a.e
#SBATCH -p gh
#SBATCH -N 5
#SBATCH --ntasks-per-node=1
#SBATCH -t 4:00:00
#SBATCH -A ASC25102
#SBATCH --array=0-1

module load cuda/12.4

set -e
cd $SCRATCH
eval "$(conda shell.bash hook)"
conda activate bb_audit_dpsgd
cd bb-audit-dpsgd

mkdir -p status

DATA_NAME="purchase"
MODEL_NAME="mlp"
N_DF=153801
LR=10
BATCH_SIZE=12143
BLOCK_SIZE=12143
N_REPS=400
N_EPOCHS=100
SEED=0
EPSILON=10
DELTA=1e-5
MAX_GRAD_NORM=1.0
DEFENSE_K=5
DEFENSE_SCORE_FN="grad_norm_unclipped"
DEFENSE_SCORE_NORM="linf"

OUTPUT_DIR="purchase_random_input_eps10"
CANARY_PATH="${OUTPUT_DIR}/random_input_canary.npy"

mkdir -p "${OUTPUT_DIR}"

# Generate a random binary input vector in Purchase feature space (600 dims).
# Both array tasks do this; it's deterministic (seed=42) so both write the same file.
python -c "
import numpy as np
np.random.seed(42)
x = np.random.randint(0, 2, size=(1, 600)).astype(np.float32)
np.save('${CANARY_PATH}', x)
print(f'Canary: shape={x.shape}, mean={x.mean():.3f}, path=${CANARY_PATH}')
"

experiments=(
    "false|${OUTPUT_DIR}/no_defense"
    "true|${OUTPUT_DIR}/defense"
)

IFS='|' read -r DEFENSE OUT_PATH <<< "${experiments[$SLURM_ARRAY_TASK_ID]}"

DEFENSE_ARGS=""
if [ "${DEFENSE}" = "true" ]; then
    DEFENSE_ARGS="--defense --defense_k ${DEFENSE_K} --defense_score_fn ${DEFENSE_SCORE_FN} --defense_score_norm ${DEFENSE_SCORE_NORM}"
fi

echo "=========================================="
echo "Task ${SLURM_ARRAY_TASK_ID}: DEFENSE=${DEFENSE}"
echo "Dataset: ${DATA_NAME}  Model: ${MODEL_NAME}  ε=${EPSILON}"
echo "Output: ${OUT_PATH}"
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
    --n_df '"${N_DF}"' \
    --n_reps '"${N_REPS}"' \
    --n_epochs '"${N_EPOCHS}"' \
    --lr '"${LR}"' \
    --batch_size '"${BATCH_SIZE}"' \
    --block_size '"${BLOCK_SIZE}"' \
    --aug_mult 1 \
    --epsilon '"${EPSILON}"' \
    --delta '"${DELTA}"' \
    --max_grad_norm '"${MAX_GRAD_NORM}"' \
    --sampling poisson \
    --holdout_audit \
    --target_type '"${CANARY_PATH}"' \
    --seed '"${SEED}"' \
    --fixed_init \
    '"${DEFENSE_ARGS}"' \
    --out '"${OUT_PATH}"''

echo "Task ${SLURM_ARRAY_TASK_ID} completed: ${OUT_PATH}"
