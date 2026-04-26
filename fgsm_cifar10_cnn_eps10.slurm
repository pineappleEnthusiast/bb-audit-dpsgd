#!/bin/bash
#SBATCH -J fgsm_cifar10_cnn_eps10
#SBATCH -o status/fgsm_cifar10_cnn_eps10_%A_%a.o
#SBATCH -e status/fgsm_cifar10_cnn_eps10_%A_%a.e
#SBATCH -p gh
#SBATCH -N 10
#SBATCH --ntasks-per-node=1
#SBATCH -t 4:00:00
#SBATCH -A ASC25102
#SBATCH --array=0-1

module load cuda/12.4

set -e
cd $WORK
eval "$(conda shell.bash hook)"
conda activate bb_audit_dpsgd
cd bb-audit-dpsgd

mkdir -p status

# -----------------------------------------------------------------------
# Fixed hyperparameters
# -----------------------------------------------------------------------
DATA_NAME="cifar10"
MODEL_NAME="cnn"
LR=3
BATCH_SIZE=3125
BLOCK_SIZE=3125
AUG_MULT=1
N_REPS=400
N_EPOCHS=100
MAX_GRAD_NORM=1
EPSILON=10.0
DELTA=1e-5
SEED=0
TARGET_TYPE="fgsm"
DEFENSE_K=5
DEFENSE_SCORE_FN="norm_x_dir_uniqueness"
DEFENSE_SCORE_NORM="linf"
PARENT_OUT="fgsm_cifar10_cnn_eps10"

mkdir -p ${PARENT_OUT}

# -----------------------------------------------------------------------
# Experiment configurations
# Format: "defense|output_name"
#   Task 0: no defense (baseline)
#   Task 1: defended with norm_x_dir_uniqueness
# -----------------------------------------------------------------------
experiments=(
    "false|cifar10_cnn_fgsm_eps10_no_defense"
    "true|cifar10_cnn_fgsm_eps10_defense_norm_x_dir_uniqueness"
)

IFS='|' read -r DEFENSE OUT_NAME <<< "${experiments[$SLURM_ARRAY_TASK_ID]}"

echo "=========================================="
echo "Job Array Task ID : ${SLURM_ARRAY_TASK_ID}"
echo "Dataset           : ${DATA_NAME}"
echo "Model             : ${MODEL_NAME}"
echo "Target type       : ${TARGET_TYPE}"
echo "Epsilon           : ${EPSILON}"
echo "Defense enabled   : ${DEFENSE}"
echo "Score function    : ${DEFENSE_SCORE_FN}"
echo "Output name       : ${OUT_NAME}"
echo "=========================================="

# Build defense flags conditionally
if [ "${DEFENSE}" = "true" ]; then
    DEFENSE_ARGS="--defense --defense_k ${DEFENSE_K} --defense_score_fn ${DEFENSE_SCORE_FN} --defense_score_norm ${DEFENSE_SCORE_NORM}"
else
    DEFENSE_ARGS=""
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
    --out '"${PARENT_OUT}/${OUT_NAME}"''

echo "=========================================="
echo "Task ${SLURM_ARRAY_TASK_ID} completed: ${OUT_NAME}"
echo "Results saved in: ${PARENT_OUT}/${OUT_NAME}/"
echo "=========================================="
