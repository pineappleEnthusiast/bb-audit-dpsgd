#!/bin/bash
#SBATCH -J ablation_global_filter
#SBATCH -o status/ablation_global_filter_%A_%a.o
#SBATCH -e status/ablation_global_filter_%A_%a.e
#SBATCH -p gh
#SBATCH -N 5
#SBATCH --ntasks-per-node=1
#SBATCH -t 5:00:00
#SBATCH -A ASC25081
#SBATCH --array=0-2

module load cuda/12.4

set -e
cd $WORK
eval "$(conda shell.bash hook)"
conda activate bb_audit_dpsgd
cd bb-audit-dpsgd

mkdir -p status

# -------------------------------------------------------
# Ablation: global top-k filtering (drop top-k highest-
# scoring samples across the entire dataset each epoch,
# regardless of class).
#
# Comparison point: the per-class defense in
# parallel_audit_model.py uses defense_k=5 per class;
# with 10 CIFAR-10 classes that drops 50 samples/epoch
# globally. So k=50 here is the direct equivalent.
#
# Array index => global k:
#   0 => 10
#   1 => 25
#   2 => 50
# -------------------------------------------------------

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
DEFENSE_SCORE_FN="grad_norm_unclipped"
DEFENSE_SCORE_NORM="linf"
PARENT_OUT="ablation_global_filter"

mkdir -p ${PARENT_OUT}

k_values=(10 25 50)
K=${k_values[$SLURM_ARRAY_TASK_ID]}
OUT_NAME="cifar10_cnn_global_k${K}"

echo "=========================================="
echo "Task ${SLURM_ARRAY_TASK_ID}: ${OUT_NAME}"
echo "Dataset: ${DATA_NAME}  Model: ${MODEL_NAME}"
echo "Epsilon: ${EPSILON}  Global filter k: ${K}"
echo "=========================================="

srun --ntasks=$SLURM_NTASKS --nodes=$SLURM_JOB_NUM_NODES bash -c '
  MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1);
  MASTER_PORT=$((29500 + '"${SLURM_ARRAY_TASK_ID}"'));
  torchrun --nnodes=$SLURM_JOB_NUM_NODES --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    parallel_audit_model_global_filter.py \
    --sampling poisson \
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
    --holdout_audit \
    --defense \
    --defense_k '"${K}"' \
    --defense_score_fn '"${DEFENSE_SCORE_FN}"' \
    --defense_score_norm '"${DEFENSE_SCORE_NORM}"' \
    --out '"${PARENT_OUT}/${OUT_NAME}"''

echo "Task ${SLURM_ARRAY_TASK_ID} completed: ${OUT_NAME}"
