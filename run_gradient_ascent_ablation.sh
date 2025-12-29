#!/bin/bash
#SBATCH -J grad_ascent_ablation
#SBATCH -o grad_ascent_ablation.o%j
#SBATCH -e grad_ascent_ablation.e%j
#SBATCH -p gh
#SBATCH -N 10
#SBATCH -n 10
#SBATCH --ntasks-per-node=1
#SBATCH -t 24:00:00
#SBATCH -A ASC25081
#SBATCH --mail-user=saloni.a.modi@utexas.edu

module load cuda/12.4

set -e
cd $SCRATCH
eval "$(conda shell.bash hook)"
conda activate bb_audit_dpsgd
cd bb-audit-dpsgd

# Common settings
N_REPS=400
N_EPOCHS=100
TARGET_TYPE="blank"
DELTA=1e-5
DEFENSE_K=5
DEFENSE_SCORE_FN="grad_norm"
DEFENSE_SCORE_NORM="l2"
MAX_GRAD_NORM=1.0

#############################################
# MNIST Experiments
#############################################
echo "=========================================="
echo "MNIST + CNN Experiments"
echo "=========================================="

DATA_NAME="mnist"
MODEL_NAME="cnn"
LR=3
EPSILON=2.0
BATCH_SIZE=4000
BLOCK_SIZE=4000

# Condition 1: NO DEFENSE
echo "Running MNIST - No defense - ε=$EPSILON..."
srun --ntasks=$SLURM_NTASKS --nodes=$SLURM_JOB_NUM_NODES bash -c '
  MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1);
  MASTER_PORT=29500;
  RANK=$SLURM_PROCID;
  torchrun --nnodes=$SLURM_JOB_NUM_NODES --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    parallel_audit_model.py \
    --data_name mnist \
    --model_name cnn \
    --n_reps 400 \
    --n_epochs 100 \
    --lr 3 \
    --max_grad_norm 1 \
    --epsilon 2.0 \
    --delta 1e-5 \
    --target_type blank \
    --batch_size 4000 \
    --block_size 4000 \
    --seed 0 \
    --holdout_audit \
    --out mnist_no_defense'

echo "✓ MNIST - No defense completed"

# Condition 2: DEFENSE WITHOUT GRADIENT ASCENT
echo "Running MNIST - Defense (no ascent) - ε=$EPSILON..."
srun --ntasks=$SLURM_NTASKS --nodes=$SLURM_JOB_NUM_NODES bash -c '
  MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1);
  MASTER_PORT=29500;
  RANK=$SLURM_PROCID;
  torchrun --nnodes=$SLURM_JOB_NUM_NODES --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    parallel_audit_model.py \
    --data_name mnist \
    --model_name cnn \
    --n_reps 400 \
    --n_epochs 100 \
    --lr 3 \
    --max_grad_norm 1 \
    --epsilon 2.0 \
    --delta 1e-5 \
    --target_type blank \
    --batch_size 4000 \
    --block_size 4000 \
    --seed 0 \
    --defense \
    --defense_k 5 \
    --defense_score_fn grad_norm \
    --defense_score_norm l2 \
    --holdout_audit \
    --out mnist_defense_no_ascent'

echo "✓ MNIST - Defense (no ascent) completed"

# Condition 3: DEFENSE WITH GRADIENT ASCENT
echo "Running MNIST - Defense (with ascent) - ε=$EPSILON..."
srun --ntasks=$SLURM_NTASKS --nodes=$SLURM_JOB_NUM_NODES bash -c '
  MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1);
  MASTER_PORT=29500;
  RANK=$SLURM_PROCID;
  torchrun --nnodes=$SLURM_JOB_NUM_NODES --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    parallel_audit_model.py \
    --data_name mnist \
    --model_name cnn \
    --n_reps 400 \
    --n_epochs 100 \
    --lr 3 \
    --max_grad_norm 1 \
    --epsilon 2.0 \
    --delta 1e-5 \
    --target_type blank \
    --batch_size 4000 \
    --block_size 4000 \
    --seed 0 \
    --defense \
    --defense_k 5 \
    --defense_apply_ascent \
    --defense_score_fn grad_norm \
    --defense_score_norm l2 \
    --holdout_audit \
    --out mnist_defense_with_ascent'

echo "✓ MNIST - Defense (with ascent) completed"

#############################################
# CIFAR-10 Experiments
#############################################
echo "=========================================="
echo "CIFAR-10 + CNN Experiments"
echo "=========================================="

DATA_NAME="cifar10"
MODEL_NAME="cnn"
LR=0.5
EPSILON=10.0
BATCH_SIZE=3125
BLOCK_SIZE=3125

# Condition 1: NO DEFENSE
echo "Running CIFAR-10 - No defense - ε=$EPSILON..."
srun --ntasks=$SLURM_NTASKS --nodes=$SLURM_JOB_NUM_NODES bash -c '
  MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1);
  MASTER_PORT=29500;
  RANK=$SLURM_PROCID;
  torchrun --nnodes=$SLURM_JOB_NUM_NODES --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    parallel_audit_model.py \
    --data_name cifar10 \
    --model_name cnn \
    --n_reps 400 \
    --n_epochs 100 \
    --lr 0.5 \
    --max_grad_norm 1 \
    --epsilon 10.0 \
    --delta 1e-5 \
    --target_type blank \
    --batch_size 3125 \
    --block_size 3125 \
    --seed 0 \
    --holdout_audit \
    --out cifar10_no_defense'

echo "✓ CIFAR-10 - No defense completed"

# Condition 2: DEFENSE WITHOUT GRADIENT ASCENT
echo "Running CIFAR-10 - Defense (no ascent) - ε=$EPSILON..."
srun --ntasks=$SLURM_NTASKS --nodes=$SLURM_JOB_NUM_NODES bash -c '
  MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1);
  MASTER_PORT=29500;
  RANK=$SLURM_PROCID;
  torchrun --nnodes=$SLURM_JOB_NUM_NODES --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    parallel_audit_model.py \
    --data_name cifar10 \
    --model_name cnn \
    --n_reps 400 \
    --n_epochs 100 \
    --lr 0.5 \
    --max_grad_norm 1 \
    --epsilon 10.0 \
    --delta 1e-5 \
    --target_type blank \
    --batch_size 3125 \
    --block_size 3125 \
    --seed 0 \
    --defense \
    --defense_k 5 \
    --defense_score_fn grad_norm \
    --defense_score_norm l2 \
    --holdout_audit \
    --out cifar10_defense_no_ascent'

echo "✓ CIFAR-10 - Defense (no ascent) completed"

# Condition 3: DEFENSE WITH GRADIENT ASCENT
echo "Running CIFAR-10 - Defense (with ascent) - ε=$EPSILON..."
srun --ntasks=$SLURM_NTASKS --nodes=$SLURM_JOB_NUM_NODES bash -c '
  MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1);
  MASTER_PORT=29500;
  RANK=$SLURM_PROCID;
  torchrun --nnodes=$SLURM_JOB_NUM_NODES --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    parallel_audit_model.py \
    --data_name cifar10 \
    --model_name cnn \
    --n_reps 400 \
    --n_epochs 100 \
    --lr 0.5 \
    --max_grad_norm 1 \
    --epsilon 10.0 \
    --delta 1e-5 \
    --target_type blank \
    --batch_size 3125 \
    --block_size 3125 \
    --seed 0 \
    --defense \
    --defense_k 5 \
    --defense_apply_ascent \
    --defense_score_fn grad_norm \
    --defense_score_norm l2 \
    --holdout_audit \
    --out cifar10_defense_with_ascent'

echo "✓ CIFAR-10 - Defense (with ascent) completed"

#############################################
# Purchase Experiments
#############################################
echo "=========================================="
echo "Purchase + MLP Experiments"
echo "=========================================="

DATA_NAME="purchase"
MODEL_NAME="mlp"
LR=0.5
EPSILON=6.0
BATCH_SIZE=12143
BLOCK_SIZE=12143

# Condition 1: NO DEFENSE
echo "Running Purchase - No defense - ε=$EPSILON..."
srun --ntasks=$SLURM_NTASKS --nodes=$SLURM_JOB_NUM_NODES bash -c '
  MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1);
  MASTER_PORT=29500;
  RANK=$SLURM_PROCID;
  torchrun --nnodes=$SLURM_JOB_NUM_NODES --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    parallel_audit_model.py \
    --data_name purchase \
    --model_name mlp \
    --n_reps 400 \
    --n_epochs 100 \
    --lr 0.5 \
    --max_grad_norm 1 \
    --epsilon 6.0 \
    --delta 1e-5 \
    --target_type blank \
    --batch_size 12143 \
    --block_size 12143 \
    --seed 0 \
    --holdout_audit \
    --out purchase_no_defense'

echo "✓ Purchase - No defense completed"

# Condition 2: DEFENSE WITHOUT GRADIENT ASCENT
echo "Running Purchase - Defense (no ascent) - ε=$EPSILON..."
srun --ntasks=$SLURM_NTASKS --nodes=$SLURM_JOB_NUM_NODES bash -c '
  MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1);
  MASTER_PORT=29500;
  RANK=$SLURM_PROCID;
  torchrun --nnodes=$SLURM_JOB_NUM_NODES --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    parallel_audit_model.py \
    --data_name purchase \
    --model_name mlp \
    --n_reps 400 \
    --n_epochs 100 \
    --lr 0.5 \
    --max_grad_norm 1 \
    --epsilon 6.0 \
    --delta 1e-5 \
    --target_type blank \
    --batch_size 12143 \
    --block_size 12143 \
    --seed 0 \
    --defense \
    --defense_k 5 \
    --defense_score_fn grad_norm \
    --defense_score_norm l2 \
    --holdout_audit \
    --out purchase_defense_no_ascent'

echo "✓ Purchase - Defense (no ascent) completed"

# Condition 3: DEFENSE WITH GRADIENT ASCENT
echo "Running Purchase - Defense (with ascent) - ε=$EPSILON..."
srun --ntasks=$SLURM_NTASKS --nodes=$SLURM_JOB_NUM_NODES bash -c '
  MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1);
  MASTER_PORT=29500;
  RANK=$SLURM_PROCID;
  torchrun --nnodes=$SLURM_JOB_NUM_NODES --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    parallel_audit_model.py \
    --data_name purchase \
    --model_name mlp \
    --n_reps 400 \
    --n_epochs 100 \
    --lr 0.5 \
    --max_grad_norm 1 \
    --epsilon 6.0 \
    --delta 1e-5 \
    --target_type blank \
    --batch_size 12143 \
    --block_size 12143 \
    --seed 0 \
    --defense \
    --defense_k 5 \
    --defense_apply_ascent \
    --defense_score_fn grad_norm \
    --defense_score_norm l2 \
    --holdout_audit \
    --out purchase_defense_with_ascent'

echo "✓ Purchase - Defense (with ascent) completed"

echo "=========================================="
echo "All Experiments Completed!"
echo "=========================================="
echo "Summary:"
echo "  - MNIST:    3 experiments at ε=2  (400 reps each, holdout audit)"
echo "  - CIFAR-10: 3 experiments at ε=10 (400 reps each, holdout audit)"
echo "  - Purchase: 3 experiments at ε=6  (400 reps each, holdout audit)"
echo "  - Total: 9 experiments on full datasets"
echo "Next steps: python analyze_gradient_ascent_ablation.py"
