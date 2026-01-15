#!/bin/bash
#SBATCH -J gen_grad_rot_canary
#SBATCH -o gen_grad_rot_canary.o%j
#SBATCH -e gen_grad_rot_canary.e%j
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH -t 01:00:00
#SBATCH -A ASC25081
#SBATCH --mail-user=srivibalaji@utexas.edu

module load cuda/12.4

set -e
cd $WORK
eval "$(conda shell.bash hook)"
cd bb-audit-dpsgd
conda activate bb_audit_dpsgd

echo "Running generate_gradient_rotation_canary_2.py..."
python generate_gradient_rotation_canary_2.py --data_name mnist --model_name cnn --num_iterations 100

echo "Done."
