#!/bin/bash
#SBATCH -J pa_mnist_canary
#SBATCH -o pa_mnist_canary.o%j
#SBATCH -e pa_mnist_canary.e%j
#SBATCH -p gh
#SBATCH -N 5
#SBATCH -n 5
#SBATCH --ntasks-per-node=1
#SBATCH -t 10:00:00
#SBATCH -A ASC25081
#SBATCH --mail-user=saloni.a.modi@utexas.edu

module load cuda/12.4

set -e
cd $SCRATCH
eval "$(conda shell.bash hook)"
conda activate bb_audit_dpsgd
cd bb-audit-dpsgd

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
    --aug_mult 1 \
    --n_reps 200 \
    --n_epochs 100 \
    --lr 3 \
    --batch_size 4000 \
    --block_size 4000 \
    --seed 0 \
    --out defense_aware_canary_mnist_blank9_lr3_bs4000_no_defense \
    --epsilon 10 \
    --delta 1e-5 \
    --max_grad_norm 1 \
    --canary_pt mnist_cnn_blank9_lr3_bs4000.pt
 '
