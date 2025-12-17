#!/bin/bash
#SBATCH -J sweep_hp              # Job name
#SBATCH -o sweep_hp.o%j          # Output file (%j = job ID)
#SBATCH -e sweep_hp.e%j          # Error file
#SBATCH -p gpu-a100                  # Partition (queue) name
#SBATCH -N 1                     # Number of nodes
#SBATCH -n 1                     # Number of tasks (MPI ranks)
#SBATCH -t 20:00:00              # Run time (hh:mm:ss)
#SBATCH -A CCR23048      # Allocation/Project ID (required on TACC)
#SBATCH --mail-user=saloni.a.modi@utexas.edu

# Print some info
echo "Running on host $(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

set -e

cd $SCRATCH
eval "$(conda shell.bash hook)"
conda activate bb_audit_dpsgd

cd bb-audit-dpsgd
# python sweep_aug_mult.py

python opacus_audit.py --data_name cifar10 --model_name cnn --aug_mult 1 --n_reps 2 --n_epochs 100 --lr 0.5 --batch_size 4000 --optimizer sgd --seed 0 --out debug --epsilon 10.0 --max_grad_norm 1 --max_physical_batch_size 1000 --fit_world_only in
python opacus_audit.py --data_name cifar10 --model_name cnn --aug_mult 1 --n_reps 2 --n_epochs 100 --lr 1 --batch_size 4000 --optimizer sgd --seed 0 --out debug --epsilon 10.0 --max_grad_norm 1 --max_physical_batch_size 1000 --fit_world_only in

python opacus_audit.py --data_name cifar10 --model_name cnn --aug_mult 4 --n_reps 2 --n_epochs 100 --lr 0.5 --batch_size 4000 --optimizer sgd --seed 0 --out debug --epsilon 10.0 --max_grad_norm 1 --max_physical_batch_size 1000 --fit_world_only in
python opacus_audit.py --data_name cifar10 --model_name cnn --aug_mult 4 --n_reps 2 --n_epochs 100 --lr 1 --batch_size 4000 --optimizer sgd --seed 0 --out debug --epsilon 10.0 --max_grad_norm 1 --max_physical_batch_size 1000 --fit_world_only in
