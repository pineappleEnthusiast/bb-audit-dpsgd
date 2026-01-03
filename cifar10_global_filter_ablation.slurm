#!/bin/bash
#SBATCH -J cifar10_global_filter
#SBATCH -o status/cifar10_global_filter_%A_%a.o
#SBATCH -e status/cifar10_global_filter_%A_%a.e
#SBATCH -p gh
#SBATCH -N 20
#SBATCH --ntasks-per-node=1
#SBATCH -t 4:00:00
#SBATCH -A ASC25081
#SBATCH --array=0-2

module load cuda/12.4

set -e
cd $SCRATCH
eval "$(conda shell.bash hook)"
conda activate bb_audit_dpsgd
cd bb-audit-dpsgd

# Common hyperparameters for CIFAR-10
DATA_NAME="cifar10"
MODEL_NAME="cnn"
N_REPS=400
N_EPOCHS=100
LR=3
MAX_GRAD_NORM=1
EPSILON=10.0
DELTA=1e-5
BATCH_SIZE=3125
BLOCK_SIZE=3125
SEED=0
TARGET_TYPE="blank"
SCORE_FN="gradient_kurtosis"
SCORE_NORM="linf"

# Parent output folder
PARENT_OUT="cifar10_global_filter_ablation"
mkdir -p ${PARENT_OUT}

# Define k values for global filtering
k_values=(10 25 50)

# Get current k value
DEFENSE_K=${k_values[$SLURM_ARRAY_TASK_ID]}
OUT_NAME="cifar10_global_k${DEFENSE_K}"

echo "=========================================="
echo "Job Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Dataset: ${DATA_NAME}"
echo "Model: ${MODEL_NAME}"
echo "Epsilon: ${EPSILON}"
echo "Scoring Function: ${SCORE_FN}"
echo "Scoring Norm: ${SCORE_NORM}"
echo "Defense K (global): ${DEFENSE_K}"
echo "Output: ${OUT_NAME}"
echo "=========================================="

# Run the audit with GLOBAL filtering
srun --ntasks=$SLURM_NTASKS --nodes=$SLURM_JOB_NUM_NODES bash -c '
  MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1);
  MASTER_PORT=$((29500 + '"${SLURM_ARRAY_TASK_ID}"'));
  RANK=$SLURM_PROCID;
  torchrun --nnodes=$SLURM_JOB_NUM_NODES --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    parallel_audit_model_global_filter.py \
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
    --seed '"${SEED}"' \
    --fixed_init \
    --aug_mult 1 \
    --defense \
    --defense_k '"${DEFENSE_K}"' \
    --defense_score_fn '"${SCORE_FN}"' \
    --defense_score_norm '"${SCORE_NORM}"' \
    --defense_apply_ascent \
    --holdout_audit \
    --out '"${PARENT_OUT}"'/'"${OUT_NAME}"''

echo "=========================================="
echo "Task ${SLURM_ARRAY_TASK_ID} completed: ${OUT_NAME}"
echo "Results saved in: ${PARENT_OUT}/${OUT_NAME}/"
echo "=========================================="
