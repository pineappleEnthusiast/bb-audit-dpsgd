#!/bin/bash
#SBATCH -J cifar10_multi_canary_defense
#SBATCH -o status/cifar10_multi_canary_defense_%A_%a.o
#SBATCH -e status/cifar10_multi_canary_defense_%A_%a.e
#SBATCH -p gh
#SBATCH -N 10
#SBATCH --ntasks-per-node=1
#SBATCH -t 20:00:00
#SBATCH -A ASC25102

module load cuda/12.0
module load nccl
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bb_audit_dpsgd

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
N_CANARIES=500

# Parent output folder
PARENT_OUT="multi_canary_defense_comparison"
mkdir -p ${PARENT_OUT}

echo "=========================================="
echo "Running CIFAR-10 Multi-Canary Audit"
echo "Dataset: ${DATA_NAME}"
echo "Model: ${MODEL_NAME}"
echo "Epsilon: ${EPSILON}"
echo "N_reps: ${N_REPS}"
echo "N_canaries: ${N_CANARIES}"
echo "=========================================="

# Condition 1: NO DEFENSE
echo ""
echo "Running CIFAR-10 - No defense - ε=${EPSILON}..."
srun --ntasks=$SLURM_NTASKS --nodes=$SLURM_JOB_NUM_NODES bash -c '
  MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1);
  MASTER_PORT=29500;
  RANK=$SLURM_PROCID;
  torchrun --nnodes=$SLURM_JOB_NUM_NODES --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    parallel_audit_multi_canary.py \
    --data_name '"${DATA_NAME}"' \
    --model_name '"${MODEL_NAME}"' \
    --n_reps '"${N_REPS}"' \
    --fixed_init \
    --holdout_audit \
    --n_epochs '"${N_EPOCHS}"' \
    --lr '"${LR}"' \
    --max_grad_norm '"${MAX_GRAD_NORM}"' \
    --epsilon '"${EPSILON}"' \
    --delta '"${DELTA}"' \
    --target_type '"${TARGET_TYPE}"' \
    --batch_size '"${BATCH_SIZE}"' \
    --block_size '"${BLOCK_SIZE}"' \
    --seed '"${SEED}"' \
    --n_canaries '"${N_CANARIES}"' \
    --aug_mult 1 \
    --out '"${PARENT_OUT}"'/cifar10_no_defense'

echo "No defense run completed"
echo ""

# Condition 2: WITH DEFENSE (gradient norm + linf norm + gradient ascent)
echo "Running CIFAR-10 - With defense - ε=${EPSILON}..."
srun --ntasks=$SLURM_NTASKS --nodes=$SLURM_JOB_NUM_NODES bash -c '
  MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1);
  MASTER_PORT=29501;
  RANK=$SLURM_PROCID;
  torchrun --nnodes=$SLURM_JOB_NUM_NODES --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    parallel_audit_multi_canary.py \
    --data_name '"${DATA_NAME}"' \
    --model_name '"${MODEL_NAME}"' \
    --n_reps '"${N_REPS}"' \
    --fixed_init \
    --holdout_audit \
    --n_epochs '"${N_EPOCHS}"' \
    --lr '"${LR}"' \
    --max_grad_norm '"${MAX_GRAD_NORM}"' \
    --epsilon '"${EPSILON}"' \
    --delta '"${DELTA}"' \
    --target_type '"${TARGET_TYPE}"' \
    --batch_size '"${BATCH_SIZE}"' \
    --block_size '"${BLOCK_SIZE}"' \
    --seed '"${SEED}"' \
    --n_canaries '"${N_CANARIES}"' \
    --aug_mult 1 \
    --defense \
    --defense_k 5 \
    --defense_score_fn prediction_margin \
    --defense_apply_ascent \
    --out '"${PARENT_OUT}"'/cifar10_with_defense'

echo "Defense run completed"
echo ""

echo "=========================================="
echo "All runs completed!"
echo "Results saved in: ${PARENT_OUT}/"
echo "  - No defense: ${PARENT_OUT}/cifar10_no_defense/"
echo "  - With defense: ${PARENT_OUT}/cifar10_with_defense/"
echo "=========================================="
