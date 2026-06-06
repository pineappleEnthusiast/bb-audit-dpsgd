#!/bin/bash
#SBATCH -J bench_cifar10_cnn_am4
#SBATCH -o status/bench_cifar10_cnn_am4_%j.o
#SBATCH -e status/bench_cifar10_cnn_am4_%j.e
#SBATCH -p gh
#SBATCH -N 5
#SBATCH --ntasks-per-node=1
#SBATCH -t 0:20:00
#SBATCH -A ASC25081

module load cuda/12.4

set -e
cd $WORK
eval "$(conda shell.bash hook)"
conda activate bb_audit_dpsgd
cd bb-audit-dpsgd

mkdir -p status

# Fixed hyperparameters (from tc_cifar10_blank_cnn_augmult4.slurm)
DATA_NAME="cifar10"
MODEL_NAME="cnn"
N_REPS=10
N_EPOCHS=100
LR=3
MAX_GRAD_NORM=1
EPSILON=10.0
DELTA=1e-5
BATCH_SIZE=3125
BLOCK_SIZE=2000
AUG_MULT=4
SEED=0
TARGET_TYPE="blank"
DEFENSE_K=5
DEFENSE_SCORE_FN="grad_norm_unclipped"
DEFENSE_SCORE_NORM="linf"
PARENT_OUT="time_benchmarks/cifar10_cnn_am4_eps10_defense"

echo "=========================================="
echo "BENCHMARK: CIFAR10 CNN AugMult4 Eps10 Defense"
echo "N_REPS: ${N_REPS} (5 in-world only), N_EPOCHS: ${N_EPOCHS}"
echo "Nodes: ${SLURM_JOB_NUM_NODES:-5}"
echo "=========================================="

START_SECONDS=$SECONDS

srun --ntasks=$SLURM_NTASKS --nodes=$SLURM_JOB_NUM_NODES bash -c '
  MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1);
  MASTER_PORT=29500;
  torchrun --nnodes=$SLURM_JOB_NUM_NODES --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    parallel_audit_model.py \
    --data_name '"${DATA_NAME}"' \
    --model_name '"${MODEL_NAME}"' \
    --n_reps '"${N_REPS}"' \
    --n_epochs '"${N_EPOCHS}"' \
    --lr '"${LR}"' \
    --batch_size '"${BATCH_SIZE}"' \
    --block_size '"${BLOCK_SIZE}"' \
    --aug_mult '"${AUG_MULT}"' \
    --epsilon '"${EPSILON}"' \
    --delta '"${DELTA}"' \
    --max_grad_norm '"${MAX_GRAD_NORM}"' \
    --target_type '"${TARGET_TYPE}"' \
    --seed '"${SEED}"' \
    --fixed_init \
    --sampling poisson \
    --holdout_audit \
    --defense --defense_k '"${DEFENSE_K}"' --defense_score_fn '"${DEFENSE_SCORE_FN}"' --defense_score_norm '"${DEFENSE_SCORE_NORM}"' \
    --fit_world_only in \
    --out '"${PARENT_OUT}"''

ELAPSED=$(( SECONDS - START_SECONDS ))
MULTIPLIER=8000  # 40K total reps / 5 benchmark reps
EST_SECONDS=$(( ELAPSED * MULTIPLIER ))
EST_BUFFERED=$(( EST_SECONDS + 7200 ))
echo "=========================================="
echo "BENCHMARK COMPLETE"
echo "Elapsed time: ${ELAPSED} seconds ($(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s)"
echo "Estimated full run (x${MULTIPLIER}): ${EST_SECONDS} seconds ($(( EST_SECONDS / 3600 ))h $(( (EST_SECONDS % 3600) / 60 ))m)"
echo "Estimated full run + 2hr buffer: ${EST_BUFFERED} seconds ($(( EST_BUFFERED / 3600 ))h $(( (EST_BUFFERED % 3600) / 60 ))m)"
echo "=========================================="
