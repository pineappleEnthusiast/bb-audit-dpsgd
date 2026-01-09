#!/bin/bash
#SBATCH -J debug_mnist_gradient
#SBATCH -o status/debug_mnist_gradient_%A.o
#SBATCH -e status/debug_mnist_gradient_%A.e
#SBATCH -p gh
#SBATCH -N 5
#SBATCH --ntasks-per-node=5
#SBATCH -t 4:00:00
#SBATCH -A ASC25081

module load cuda/12.4

set -e
cd $SCRATCH
eval "$(conda shell.bash hook)"
conda activate bb_audit_dpsgd
cd bb-audit-dpsgd

# Common hyperparameters
DATA_NAME="mnist"
MODEL_NAME="cnn"
N_REPS=200
N_EPOCHS=100
LR=3
MAX_GRAD_NORM=1.0
EPSILON=10.0
DELTA=1e-5
BATCH_SIZE=4000
BLOCK_SIZE=4000
SEED=0
OUT_DIR="debug"

echo "=========================================="
echo "Dataset: ${DATA_NAME}"
echo "Model: ${MODEL_NAME}"
echo "N Reps: ${N_REPS}"
echo "N Epochs: ${N_EPOCHS}"
echo "LR: ${LR}"
echo "Max Grad Norm: ${MAX_GRAD_NORM}"
echo "Epsilon: ${EPSILON}"
echo "Delta: ${DELTA}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Block Size: ${BLOCK_SIZE}"
echo "Seed: ${SEED}"
echo "Output: ${OUT_DIR}"
echo "=========================================="

# Run the audit
srun --ntasks=$SLURM_NTASKS --nodes=$SLURM_JOB_NUM_NODES bash -c '
  MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1);
  MASTER_PORT=$((29500 + '"${SLURM_ARRAY_TASK_ID:-0}"'));
  torchrun --nnodes=$SLURM_JOB_NUM_NODES --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    parallel_audit_model.py \
    --data_name '"${DATA_NAME}"' \
    --model_name '"${MODEL_NAME}"' \
    --gradient_space_canary_pt canary_gradient.pt \
    --n_reps '"${N_REPS}"' \
    --n_epochs '"${N_EPOCHS}"' \
    --lr '"${LR}"' \
    --max_grad_norm '"${MAX_GRAD_NORM}"' \
    --epsilon '"${EPSILON}"' \
    --delta '"${DELTA}"' \
    --batch_size '"${BATCH_SIZE}"' \
    --block_size '"${BLOCK_SIZE}"' \
    --seed '"${SEED}"' \
    --fixed_init \
    --out '"${OUT_DIR}"''

echo "=========================================="
echo "Job completed: ${OUT_DIR}"
echo "Results saved in: ${OUT_DIR}/"
echo "=========================================="
