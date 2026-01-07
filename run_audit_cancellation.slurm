#!/bin/bash
#SBATCH -J cancellation_audit
#SBATCH -o cancellation_audit.o%j
#SBATCH -e cancellation_audit.e%j
#SBATCH -p gh
#SBATCH -N 5
#SBATCH -n 5
#SBATCH --ntasks-per-node=1
#SBATCH -t 3:00:00
#SBATCH -A ASC25081
#SBATCH --mail-user=srivibalaji@utexas.edu

module load cuda/12.4

set -e
cd $WORK
eval "$(conda shell.bash hook)"
conda activate bb_audit_dpsgd
cd bb-audit-dpsgd

# 1. Audit WITHOUT Defense
echo "Starting audit WITHOUT defense..."
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
    --canary_pt cancellation_canaries.pt \
    --aug_mult 1 \
    --n_reps 400 \
    --n_epochs 100 \
    --lr 3 \
    --batch_size 4000 \
    --block_size 4000 \
    --epsilon 10.0 \
    --delta 1e-5 \
    --max_grad_norm 1.0 \
    --fixed_init \
    --holdout_audit \
    --out debug/cancellation_audit/no_defense
'

# 2. Audit WITH Defense
echo "Starting audit WITH defense..."
srun --ntasks=$SLURM_NTASKS --nodes=$SLURM_JOB_NUM_NODES bash -c '
  MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1);
  MASTER_PORT=29501; # increment port for second run
  RANK=$SLURM_PROCID;
  torchrun --nnodes=$SLURM_JOB_NUM_NODES --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    parallel_audit_multi_canary.py \
    --data_name mnist \
    --model_name cnn \
    --canary_pt cancellation_canaries.pt \
    --aug_mult 1 \
    --n_reps 400 \
    --n_epochs 100 \
    --lr 3 \
    --batch_size 4000 \
    --block_size 4000 \
    --epsilon 10.0 \
    --delta 1e-5 \
    --max_grad_norm 1.0 \
    --fixed_init \
    --holdout_audit \
    --out debug/cancellation_audit/with_defense \
    --defense \
    --defense_k 5 \
    --defense_score_norm linf \
    --defense_score_fn grad_norm \
    --defense_apply_ascent
'
