#!/bin/bash
#SBATCH -J cifar10_filter_epochs
#SBATCH -o status/cifar10_filter_epochs_%A_%a.o
#SBATCH -e status/cifar10_filter_epochs_%A_%a.e
#SBATCH -p gh
#SBATCH -N 20
#SBATCH -n 20
#SBATCH --ntasks-per-node=1
#SBATCH -t 04:00:00
#SBATCH -A ASC25081
#SBATCH --mail-user=saloni.a.modi@utexas.edu
#SBATCH --array=0-3

module load cuda/12.4

set -e
cd $SCRATCH
eval "$(conda shell.bash hook)"
conda activate bb_audit_dpsgd
cd bb-audit-dpsgd

# Create parent output folder
mkdir -p cifar10_varying_filter_epochs

# NOTE: This script assumes a --defense_filter_every parameter exists in parallel_audit_model.py
# If it doesn't exist, you'll need to add it to the argparse and modify the defense logic
# to only apply filtering when (epoch % defense_filter_every == 0)

# Define experiment configurations
# Array format: "filter_every|output_name"
experiments=(
    "1|cifar10_filter_every_1_epoch_20_gpus"
    "5|cifar10_filter_every_5_epochs_20_gpus"
    "10|cifar10_filter_every_10_epochs_20_gpus"
    "20|cifar10_filter_every_20_epochs_20_gpus"
)

# Get current experiment configuration
IFS='|' read -r FILTER_EVERY OUT_NAME <<< "${experiments[$SLURM_ARRAY_TASK_ID]}"

echo "=========================================="
echo "Job Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running: $OUT_NAME"
echo "Model: CNN, Filter every: $FILTER_EVERY epochs"
echo "Epsilon: 10.0 (fixed), Defense K: 5"
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
    --defense_k 5 \
    --defense_apply_ascent \
    --defense_score_fn grad_norm \
    --defense_score_norm l2 \
    --defense_filter_every $FILTER_EVERY \
    --holdout_audit \
    --out cifar10_varying_filter_epochs/$OUT_NAME"

echo "✓ Experiment completed: $OUT_NAME"
