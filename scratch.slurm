#!/bin/bash

#SBATCH -A CCR23048
#SBATCH -J MLSEC           # Job name
#SBATCH -o status/DPSGD.o%j       # Name of stdout output file
#SBATCH -e status/DPSGD.e%j       # Name of stderr error file
#SBATCH -p gpu-a100-small       # Queue (partition) name
#SBATCH -N 1                # Total # of nodes (must be 1 for serial)
#SBATCH -n 1                # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 00:30:00         # Run time (hh:mm:ss)
#SBATCH --mail-type=all     # Send email at begin and end of job
#SBATCH --mail-user=saloni.a.modi@utexas.edu

set -e

cd $SCRATCH
eval "$(conda shell.bash hook)"
conda activate bb_audit_dpsgd

cd bb-audit-dpsgd
python3 audit_model.py --data_name mnist --model_name cnn --lr 1.33e-4 --epsilon 10 --fixed_init --out debug --block_size 10000 --target_type blank --fit_world_only in --n_reps 2 --batch_size 4096


python3 audit_model.py --data_name cifar10 --model_name wideresnet --lr 1.33e-4 --epsilon 10 --fixed_init --out debug --block_size 50 --target_type blank --fit_world_only in --n_reps 2