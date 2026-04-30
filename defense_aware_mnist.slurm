#!/bin/bash
#SBATCH -J defense_aware_mnist
#SBATCH -o status/defense_aware_mnist_%j.o
#SBATCH -e status/defense_aware_mnist_%j.e
#SBATCH -p gh
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -t 10:00:00
#SBATCH -A ASC25081

module load cuda/12.4

set -e
cd $SCRATCH
eval "$(conda shell.bash hook)"
conda activate bb_audit_dpsgd
cd bb-audit-dpsgd

mkdir -p status
mkdir -p defense_aware_canaries/mnist

python defense_aware_canary.py \
    --data_name mnist --model_name cnn \
    --n_epochs 100 --lr 3 --batch_size 4000 \
    --epsilon 10.0 --delta 1e-5 --max_grad_norm 1.0 \
    --defense_k 5 \
    --bilevel --n_outer 50 --n_inner 3 --n_runs 3 \
    --lam 0.0 0.01 0.1 0.5 1.0 5.0 10.0 100.0 \
    --source_classes 1 \
    --seed 0 \
    --output defense_aware_canaries/mnist/canary.pt
