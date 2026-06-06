#!/bin/bash
#SBATCH -J mnist_sanity_check
#SBATCH -o status/mnist_sanity_check_%j.o
#SBATCH -e status/mnist_sanity_check_%j.e
#SBATCH -p gh
#SBATCH -N 5
#SBATCH --ntasks-per-node=1
#SBATCH -t 0:20:00
#SBATCH -A ASC25081

module load cuda/12.4

set -e
cd $WORK
eval "$(conda shell.bash hook)"
conda activate bb_audit_dpsgd
cd bb-audit-dpsgd

mkdir -p status

# Fixed hyperparameters
DATA_NAME="mnist"
MODEL_NAME="cnn"
N_REPS=10
N_EPOCHS=10
LR=3
MAX_GRAD_NORM=1
EPSILON=10.0
DELTA=1e-5
BATCH_SIZE=4000
BLOCK_SIZE=4000
SEED=0
TARGET_TYPE="blank"
PARENT_OUT="mnist_sanity_check"

echo "=========================================="
echo "Sanity check: MNIST with ${N_REPS} reps on ${SLURM_JOB_NUM_NODES} nodes"
echo "Dataset: ${DATA_NAME}  Model: ${MODEL_NAME}"
echo "N_REPS: ${N_REPS}  Epsilon: ${EPSILON}"
echo "=========================================="

srun --ntasks=$SLURM_NTASKS --nodes=$SLURM_JOB_NUM_NODES bash -c '
  MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1);
  MASTER_PORT=29500;
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
    --aug_mult 1 \
    --epsilon '"${EPSILON}"' \
    --delta '"${DELTA}"' \
    --max_grad_norm '"${MAX_GRAD_NORM}"' \
    --target_type '"${TARGET_TYPE}"' \
    --seed '"${SEED}"' \
    --fixed_init \
    --sampling poisson \
    --holdout_audit \
    --out '"${PARENT_OUT}"''

echo "Sanity check completed!"
