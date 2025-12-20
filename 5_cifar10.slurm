#!/bin/bash
#SBATCH -J pa_c10_5gpu
#SBATCH -o pa_c10_5gpu.o%j
#SBATCH -e pa_c10_5gpu.e%j
#SBATCH -p gh
#SBATCH -N 5
#SBATCH -n 5
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH -t 00:30:00
#SBATCH -A ASC25081
#SBATCH --mail-user=saloni.a.modi@utexas.edu

module load cuda/12.4

set -e
cd $SCRATCH
eval "$(conda shell.bash hook)"
conda activate bb_audit_dpsgd
cd bb-audit-dpsgd

srun --ntasks=$SLURM_NNODES --nodes=$SLURM_NNODES bash -c '
  MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1);
  MASTER_PORT=29500;
  RANK=$SLURM_PROCID;
  torchrun --nnodes=$SLURM_NNODES --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    parallel_audit_model.py \
    --data_name cifar10 \
    --model_name cnn \
    --n_reps 10 \
    --n_epochs 100 \
    --lr 3 \
    --batch_size 3125 \
    --block_size 3125 \
    --seed 0 \
    --epsilon 10 \
    --delta 1e-5 \
    --max_grad_norm 1 \
    --aug_mult 1 \
    --fit_world_only in \
    --out debug'