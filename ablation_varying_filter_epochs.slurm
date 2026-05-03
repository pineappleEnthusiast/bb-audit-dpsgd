#!/bin/bash
#SBATCH -J ablation_filter_epochs
#SBATCH -o status/ablation_filter_epochs_%A_%a.o
#SBATCH -e status/ablation_filter_epochs_%A_%a.e
#SBATCH -p gh
#SBATCH -N 5
#SBATCH --ntasks-per-node=1
#SBATCH -t 5:00:00
#SBATCH -A ASC25081
#SBATCH --array=0-3

module load cuda/12.4

set -e
cd $WORK
eval "$(conda shell.bash hook)"
conda activate bb_audit_dpsgd
cd bb-audit-dpsgd

mkdir -p status

# -------------------------------------------------------
# Experiment: sensitivity of the defense to filtering frequency
# (filtering applied once every k epochs instead of every epoch)
# Array index => filter_every value:
#   0 => 1   (every epoch — baseline)
#   1 => 5
#   2 => 10
#   3 => 20
# -------------------------------------------------------

# Fixed hyperparameters — match the standard MNIST CNN setup
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
TARGET_TYPE="blank"
DEFENSE_K=5
DEFENSE_SCORE_FN="grad_norm_unclipped"
DEFENSE_SCORE_NORM="linf"
PARENT_OUT="ablation_varying_filter_epochs"

# Map array task ID to filter_every value
filter_every_values=(1 5 10 20)
FILTER_EVERY=${filter_every_values[$SLURM_ARRAY_TASK_ID]}
OUT_NAME="cifar10_cnn_defense_filter_every${FILTER_EVERY}"

echo "=========================================="
echo "Task ${SLURM_ARRAY_TASK_ID}: ${OUT_NAME}"
echo "Dataset: ${DATA_NAME}  Model: ${MODEL_NAME}"
echo "Epsilon: ${EPSILON}  Filter every: ${FILTER_EVERY} epochs"
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
    --aug_mult '"${AUG_MULT}"' \
    --epsilon '"${EPSILON}"' \
    --delta '"${DELTA}"' \
    --max_grad_norm '"${MAX_GRAD_NORM}"' \
    --target_type '"${TARGET_TYPE}"' \
    --seed '"${SEED}"' \
    --fixed_init \
    --sampling poisson \
    --holdout_audit \
    --defense \
    --defense_k '"${DEFENSE_K}"' \
    --defense_filter_every '"${FILTER_EVERY}"' \
    --defense_score_fn '"${DEFENSE_SCORE_FN}"' \
    --defense_score_norm '"${DEFENSE_SCORE_NORM}"' \
    --out '"${PARENT_OUT}"'/'"${OUT_NAME}"''

echo "Task ${SLURM_ARRAY_TASK_ID} completed: ${OUT_NAME}"
