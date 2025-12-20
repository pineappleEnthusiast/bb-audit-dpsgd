#!/bin/bash
#SBATCH -J pa_c10_5gpu
#SBATCH -o pa_c10_5gpu.o%j
#SBATCH -e pa_c10_5gpu.e%j
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:30:00
#SBATCH -A ASC25081
#SBATCH --gres=gpu:5
#SBATCH --mail-user=saloni.a.modi@utexas.edu

module load cuda/12.4

echo "Running on host $(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

set -e

cd $SCRATCH
eval "$(conda shell.bash hook)"
conda activate bb_audit_dpsgd

cd bb-audit-dpsgd

srun torchrun --standalone --nnodes=1 --nproc_per_node=5 parallel_audit_model.py \
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
  --out debug