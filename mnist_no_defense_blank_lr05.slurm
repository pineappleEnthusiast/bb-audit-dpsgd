#!/bin/bash
#SBATCH -J mnist_no_def_blank_lr05
#SBATCH -o status/mnist_no_def_blank_lr05_%j.o
#SBATCH -e status/mnist_no_def_blank_lr05_%j.e
#SBATCH -p gh
#SBATCH -N 20
#SBATCH --ntasks-per-node=1
#SBATCH -t 1:00:00
#SBATCH -A ASC25081

module load cuda/12.4

set -e
cd $SCRATCH
eval "$(conda shell.bash hook)"
conda activate bb_audit_dpsgd
cd bb-audit-dpsgd

# Hyperparameters
DATA_NAME="mnist"
MODEL_NAME="cnn"
N_REPS=200
N_EPOCHS=100
LR=0.5
MAX_GRAD_NORM=1
EPSILON=10.0
DELTA=1e-5
BATCH_SIZE=4000
BLOCK_SIZE=4000
SEED=0
TARGET_TYPE="blank"
OUT_NAME="mnist_no_defense_blank_lr05_eps10"

echo "=========================================="
echo "Dataset: ${DATA_NAME}"
echo "Model: ${MODEL_NAME}"
echo "Epsilon: ${EPSILON}"
echo "Learning Rate: ${LR}"
echo "Reps: ${N_REPS}"
echo "Target Type: ${TARGET_TYPE}"
echo "Output: ${OUT_NAME}"
echo "=========================================="

# Run the audit (non-holdout, no --holdout_audit flag)
srun --ntasks=$SLURM_NTASKS --nodes=$SLURM_JOB_NUM_NODES bash -c '
  MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1);
  MASTER_PORT=29500;
  RANK=$SLURM_PROCID;
  torchrun --nnodes=$SLURM_JOB_NUM_NODES --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    parallel_audit_model.py \
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
    --aug_mult 1 \
    --out '"${OUT_NAME}"''

echo "=========================================="
echo "Audit completed: ${OUT_NAME}"
echo "Results saved in: ${OUT_NAME}/"
echo "=========================================="
