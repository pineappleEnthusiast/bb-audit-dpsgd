#!/bin/bash
#SBATCH -J test_purchase_blank_2              # Job name
#SBATCH -o test_purchase_blank_2.o%j          # Output file (%j = job ID)
#SBATCH -e test_purchase_blank_2.e%j          # Error file
#SBATCH -p gh                    # Partition (queue) name
#SBATCH -N 1                     # Number of nodes
#SBATCH -n 1                     # Number of tasks (MPI ranks)
#SBATCH -t 10:00:00              # Run time (hh:mm:ss)
#SBATCH -A ASC25081       # Allocation/Project ID (required on TACC)

# Load any needed modules
module load cuda/12.4

# Print some info
echo "Running on host $(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

set -e

cd $WORK
eval "$(conda shell.bash hook)"
conda activate bb_audit_dpsgd

cd bb-audit-dpsgd
# python3 audit_model.py --data_name purchase --model_name mlp --lr 5e-3 --epsilon 10 --fixed_init --out results/test_purchase_blank_2 --block_size 12288 --target_type blank --n_reps 400 --batch_size 12288 --defense
python3 audit_model.py --data_name purchase --model_name lr --lr 5e-3 --epsilon 10 --fixed_init --out results/test_purchase_blank_2 --block_size 12288 --target_type blank --n_reps 400 --batch_size 12288 --defense