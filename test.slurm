#!/bin/bash
#SBATCH -J test_gpu              # Job name
#SBATCH -o test_gpu.o%j          # Output file (%j = job ID)
#SBATCH -e test_gpu.e%j          # Error file
#SBATCH -p gh-dev                # Partition (queue) name
#SBATCH -N 1                     # Number of nodes
#SBATCH -n 1                     # Number of tasks (MPI ranks)
#SBATCH -t 00:05:00              # Run time (hh:mm:ss)
#SBATCH -A ASC25081       # Allocation/Project ID (required on TACC)

# Load any needed modules
module load cuda/12.4

# Print some info
echo "Running on host $(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# Run a GPU test (replace with your program)
nvidia-smi
