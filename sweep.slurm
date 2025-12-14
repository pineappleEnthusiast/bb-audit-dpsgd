#!/bin/bash
#SBATCH -J sweep_hp              # Job name
#SBATCH -o sweep_hp.o%j          # Output file (%j = job ID)
#SBATCH -e sweep_hp.e%j          # Error file
#SBATCH -p gh                    # Partition (queue) name
#SBATCH -N 1                     # Number of nodes
#SBATCH -n 1                     # Number of tasks (MPI ranks)
#SBATCH -t 20:00:00              # Run time (hh:mm:ss)
#SBATCH -A ASC25081       # Allocation/Project ID (required on TACC)
#SBATCH --mail-user=saloni.a.modi@utexas.edu

# Load any needed modules
module load cuda/12.4

# Print some info
echo "Running on host $(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

set -e

cd $SCRATCH
eval "$(conda shell.bash hook)"
conda activate bb_audit_dpsgd

cd bb-audit-dpsgd
python sweep_aug_mult.py