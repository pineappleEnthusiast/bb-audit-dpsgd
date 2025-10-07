#!/bin/bash
#SBATCH -J test_mnist_blank_25              # Job name
#SBATCH -o test_mnist_blank_25.o%j          # Output file (%j = job ID)
#SBATCH -e test_mnist_blank_25.e%j          # Error file
#SBATCH -p gh                    # Partition (queue) name
#SBATCH -N 1                     # Number of nodes
#SBATCH -n 1                     # Number of tasks (MPI ranks)
#SBATCH -t 06:00:00              # Run time (hh:mm:ss)
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
python3 audit_model.py --data_name mnist --model_name cnn --lr 1.33e-4 --epsilon 10 --fixed_init --out debug --block_size 4096 --target_type blank --blank_alpha 25 --n_reps 400 --batch_size 4096 --defense