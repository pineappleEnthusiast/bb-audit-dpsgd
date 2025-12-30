#!/bin/bash
#SBATCH -J cifar10_varying_k
#SBATCH -o status/cifar10_varying_k_%A_%a.o
#SBATCH -e status/cifar10_varying_k_%A_%a.e
#SBATCH -p gh
#SBATCH -N 20
#SBATCH -n 20
#SBATCH --ntasks-per-node=1
#SBATCH -t 04:00:00
#SBATCH -A ASC25081
#SBATCH --mail-user=saloni.a.modi@utexas.edu
#SBATCH --array=0-4

module load cuda/12.4

set -e
cd $SCRATCH
eval "$(conda shell.bash hook)"
conda activate bb_audit_dpsgd
cd bb-audit-dpsgd

# Create parent output folder
mkdir -p cifar10_varying_defense_bandwidth

# Define experiment configurations
# Array format: "defense_k|output_name"
experiments=(
    "1|cifar10_defense_k1_20_gpus"
    "2|cifar10_defense_k2_20_gpus"
    "3|cifar10_defense_k3_20_gpus"
    "4|cifar10_defense_k4_20_gpus"
    "5|cifar10_defense_k5_20_gpus"
)

# Get current experiment configuration
IFS='|' read -r DEFENSE_K OUT_NAME <<< "${experiments[$SLURM_ARRAY_TASK_ID]}"

echo "=========================================="
echo "Job Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running: $OUT_NAME"
echo "Model: CNN, Defense K: $DEFENSE_K"
echo "Epsilon: 10.0 (fixed)"
echo "=========================================="

# Run the experiment
srun --ntasks=$SLURM_NTASKS --nodes=$SLURM_JOB_NUM_NODES bash -c "
  MASTER_ADDR=\$(scontrol show hostnames \$SLURM_JOB_NODELIST | head -n 1);
  MASTER_PORT=29500;
  RANK=\$SLURM_PROCID;
  torchrun --nnodes=\$SLURM_JOB_NUM_NODES --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=\${MASTER_ADDR}:\${MASTER_PORT} \
    parallel_audit_model.py \
    --data_name cifar10 \
    --model_name cnn \
    --n_reps 400 \
    --n_epochs 100 \
    --lr 3 \
    --max_grad_norm 1 \
    --epsilon 10.0 \
    --delta 1e-5 \
    --target_type blank \
    --batch_size 3125 \
    --block_size 3125 \
    --seed 0 \
    --defense \
    --defense_k $DEFENSE_K \
    --defense_apply_ascent \
    --defense_score_fn grad_norm \
    --defense_score_norm l2 \
    --holdout_audit \
    --out cifar10_varying_defense_bandwidth/$OUT_NAME"

echo "✓ Experiment completed: $OUT_NAME"
