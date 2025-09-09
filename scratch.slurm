#!/bin/bash

#SBATCH -A CCR23048
#SBATCH -J MLSEC           # Job name
#SBATCH -o status/DPSGD.o%j       # Name of stdout output file
#SBATCH -e status/DPSGD.e%j       # Name of stderr error file
#SBATCH -p gpu-a100-small       # Queue (partition) name
#SBATCH -N 2                    # Number of nodes
#SBATCH --ntasks-per-node=4     # Number of tasks (processes) per node
#SBATCH --cpus-per-task=4       # Number of CPU cores per task
#SBATCH --gres=gpu:4            # Number of GPUs per node
#SBATCH -t 00:30:00             # Run time (hh:mm:ss)
#SBATCH --mail-type=all         # Send email at begin and end of job
#SBATCH --mail-user=saloni.a.modi@utexas.edu

# Load necessary modules
module load cuda/11.8
module load python3/3.10

set -e

# Set up environment
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=^docker0,lo

# Get node list
NODES=$(scontrol show hostnames $SLURM_JOB_NODELIST | tr '\n' ' ')
MASTER_NODE=$(echo $NODES | cut -d' ' -f1)
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=12345

# Create status directory if it doesn't exist
mkdir -p status

# Activate conda environment
cd $SCRATCH
eval "$(conda shell.bash hook)"
conda activate bb_audit_dpsgd

cd bb-audit-dpsgd

# First training job
echo "Starting MNIST training on $(date)"
srun --mpi=pmi2 -n $SLURM_NTASKS -N $SLURM_NNODES --ntasks-per-node=$SLURM_NTASKS_PER_NODE \
    python -m torch.distributed.run \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=$SLURM_GPUS_PER_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:12345 \
    audit_model.py \
    --data_name mnist \
    --model_name cnn \
    --lr 1.33e-4 \
    --epsilon 10 \
    --fixed_init \
    --out debug \
    --block_size 10000 \
    --target_type blank \
    --fit_world_only in \
    --n_reps 2 \
    --batch_size 4096

echo "MNIST training completed on $(date)"

# Second training job
echo "Starting CIFAR-10 training on $(date)"
srun --mpi=pmi2 -n $SLURM_NTASKS -N $SLURM_NNODES --ntasks-per-node=$SLURM_NTASKS_PER_NODE \
    python -m torch.distributed.run \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=$SLURM_GPUS_PER_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:12345 \
    audit_model.py \
    --data_name cifar10 \
    --model_name wideresnet \
    --lr 1.33e-4 \
    --epsilon 10 \
    --fixed_init \
    --out debug \
    --block_size 50 \
    --target_type blank \
    --fit_world_only in \
    --n_reps 2

echo "CIFAR-10 training completed on $(date)"