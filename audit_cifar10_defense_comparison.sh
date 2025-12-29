#!/bin/bash
#SBATCH -J cifar10_defense_comp
#SBATCH -o cifar10_defense_comp.o%j
#SBATCH -e cifar10_defense_comp.e%j
#SBATCH -p gh
#SBATCH -N 10
#SBATCH -n 10
#SBATCH --ntasks-per-node=1
#SBATCH -t 12:00:00
#SBATCH -A ASC25081
#SBATCH --mail-user=saloni.a.modi@utexas.edu

module load cuda/12.4

set -e
cd $SCRATCH
eval "$(conda shell.bash hook)"
conda activate bb_audit_dpsgd
cd bb-audit-dpsgd

# Experiment 1: No Defense
echo "Running CIFAR-10 WITHOUT Defense"
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
    --n_reps 200 \
    --n_epochs 100 \
    --lr 0.5 \
    --batch_size 2000 \
    --block_size 2000 \
    --seed 0 \
    --epsilon 10 \
    --delta 1e-5 \
    --max_grad_norm 1 \
    --aug_mult 1 \
    --target_type blank \
    --alpha 0.05 \
    --out cifar10_eps10_no_defense'

# Experiment 2: With Defense
echo "Running CIFAR-10 WITH Defense"
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
    --n_reps 200 \
    --n_epochs 100 \
    --lr 0.5 \
    --batch_size 2000 \
    --block_size 2000 \
    --seed 0 \
    --epsilon 10 \
    --delta 1e-5 \
    --max_grad_norm 1 \
    --aug_mult 1 \
    --target_type blank \
    --alpha 0.05 \
    --defense \
    --defense_k 5 \
    --defense_score_fn grad_norm \
    --defense_score_norm l2 \
    --defense_apply_ascent \
    --out cifar10_eps10_with_defense'
