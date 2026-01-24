#!/bin/bash
#SBATCH -J mnist_grad_cancel
#SBATCH -o mnist_grad_cancel.o%j
#SBATCH -e mnist_grad_cancel.e%j
#SBATCH -p gpu-a100-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH -t 20:00:00
#SBATCH -A CCR23048

module load cuda/12.4

set -e
cd $SCRATCH
eval "$(conda shell.bash hook)"
conda activate bb_audit_dpsgd
cd bb-audit-dpsgd

# Step 1: Generate gradient cancelling canaries
echo "=========================================="
echo "Step 1: Generating gradient cancelling canaries"
echo "=========================================="

python generate_gradient_cancelling_attack.py \
  --model_name cnn \
  --data_name mnist \
  --n_group_a 500 \
  --n_group_b 600 \
  --alpha 0.99 \
  --output_dir mnist_gradient_cancelling_canaries \
  --seed 0 \
  --device cuda:0 \
  --n_epochs 100 \
  --lr 3 \
  --max_grad_norm 1.0 \
  --epsilon 10.0 \
  --delta 1e-5 \
  --batch_size 4000 \
  --block_size 4000

echo "=========================================="
echo "Step 2: Running audit without defense"
echo "=========================================="

# Step 2: Run audit without defense
srun --ntasks=$SLURM_NTASKS --nodes=$SLURM_JOB_NUM_NODES bash -c '
  MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1);
  MASTER_PORT=29500;
  RANK=$SLURM_PROCID;
  torchrun --nnodes=$SLURM_JOB_NUM_NODES --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    parallel_audit_multi_canary.py \
    --data_name mnist \
    --model_name cnn \
    --target_type gradient_space_canary \
    --gradient_space_canary_pt mnist_gradient_cancelling_canaries/gradient_space_canaries.pt \
    --n_reps 400 \
    --fixed_init \
    --holdout_audit \
    --n_epochs 100 \
    --lr 3 \
    --max_grad_norm 1.0 \
    --epsilon 10.0 \
    --delta 1e-5 \
    --batch_size 4000 \
    --block_size 4000 \
    --seed 0 \
    --out mnist_gradient_cancelling_canary_no_defense'

echo "=========================================="
echo "Step 3: Running audit with defense"
echo "=========================================="

# Step 3: Run audit with defense
srun --ntasks=$SLURM_NTASKS --nodes=$SLURM_JOB_NUM_NODES bash -c '
  MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1);
  MASTER_PORT=29501;
  RANK=$SLURM_PROCID;
  torchrun --nnodes=$SLURM_JOB_NUM_NODES --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    parallel_audit_multi_canary.py \
    --data_name mnist \
    --model_name cnn \
    --target_type gradient_space_canary \
    --gradient_space_canary_pt mnist_gradient_cancelling_canaries/gradient_space_canaries.pt \
    --n_reps 400 \
    --fixed_init \
    --holdout_audit \
    --n_epochs 100 \
    --lr 3 \
    --max_grad_norm 1.0 \
    --epsilon 10.0 \
    --delta 1e-5 \
    --batch_size 4000 \
    --block_size 4000 \
    --seed 0 \
    --defense \
    --defense_apply_ascent \
    --defense_global_filter \
    --out mnist_gradient_cancelling_canary_defense'

echo "=========================================="
echo "All steps completed successfully"
echo "=========================================="
