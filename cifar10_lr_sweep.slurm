#!/bin/bash
#SBATCH -J cifar10_lr3
#SBATCH -o status/cifar10_lr3_%A.out
#SBATCH -e status/cifar10_lr3_%A.err
#SBATCH -p gh
#SBATCH -N 20
#SBATCH -n 20
#SBATCH -t 4:00:00
#SBATCH -A ASC25081

# Single learning rate
LR=3

# Output directory
OUT_DIR="results/cifar10_lr3/cifar10_cnn_eps10.0"
mkdir -p ${OUT_DIR}

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Learning Rate: ${LR}"
echo "Output Directory: ${OUT_DIR}"
echo "=========================================="

# Get master node address
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=29500

echo "Master Address: ${MASTER_ADDR}"
echo "Master Port: ${MASTER_PORT}"

# Run the audit with torchrun
srun --ntasks=20 --nodes=20 bash -c "
  export MASTER_ADDR=${MASTER_ADDR};
  export MASTER_PORT=${MASTER_PORT};
  export RANK=\$SLURM_PROCID;
  torchrun --nnodes=20 --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=\${MASTER_ADDR}:\${MASTER_PORT} \
    parallel_audit_model.py \
    --data_name cifar10 \
    --model_name cnn \
    --n_reps 400 \
    --holdout_audit \
    --n_epochs 100 \
    --lr ${LR} \
    --max_grad_norm 1.0 \
    --epsilon 10.0 \
    --delta 1e-5 \
    --target_type blank \
    --batch_size 3125 \
    --block_size 3125 \
    --fixed_init \
    --out ${OUT_DIR}
"

echo "=========================================="
echo "Job completed for LR=${LR}"
echo "=========================================="
