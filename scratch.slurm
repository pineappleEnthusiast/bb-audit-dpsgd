#!/bin/bash

#SBATCH -A CCR23048
#SBATCH -J MLSEC           # Job name
#SBATCH -o status/DPSGD.o%j       # Name of stdout output file
#SBATCH -e status/DPSGD.e%j       # Name of stderr error file
#SBATCH -p gpu-a100-small       # Queue (partition) name
#SBATCH -N 1                # Total # of nodes (must be 1 for serial)
#SBATCH -n 1                # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 04:00:00         # Run time (hh:mm:ss)
#SBATCH --mail-type=all     # Send email at begin and end of job
#SBATCH --mail-user=saloni.a.modi@utexas.edu

set -e

cd $SCRATCH
eval "$(conda shell.bash hook)"
conda activate bb_audit_dpsgd

cd bb-audit-dpsgd
# python3 audit_model.py --data_name mnist --model_name lr --epsilon 10.0 --n_reps=10 --adversarial_audit --n_epochs=10
# python3 audit_model.py --data_name mnist --model_name=cnn --epsilon 10.0 --n_reps=200 --adversarial_audit --n_epochs=5 --block_size=200
python3 audit_model.py --data_name mnist --model_name cnn --lr 1.33e-4 --epsilon 10 --fixed_init --out mnist/badnets/2/ --target_type badnets --badnets_label 2
# python3 audit_model.py --data_name mnist --model_name cnn --lr 1.33e-4 --epsilon 10 --fixed_init --out adaptive_defense/weights --defense # --resume # --cutoff 32 # --resume # model_init/fixed_average/

# python3 audit_model.py --data_name mnist --model_name cnn --lr 1.33e-4 --epsilon 10 --fixed_init --out debug --defense # --resume # model_init/fixed_average/