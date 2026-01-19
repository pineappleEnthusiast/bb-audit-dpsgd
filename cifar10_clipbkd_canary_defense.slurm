#!/bin/bash
#SBATCH -J cifar10_clipbkd_canary_defense
#SBATCH -o status/cifar10_clipbkd_canary_defense_%A.o
#SBATCH -e status/cifar10_clipbkd_canary_defense_%A.e
#SBATCH -p gh
#SBATCH -N 20
#SBATCH --ntasks-per-node=1
#SBATCH -t 4:00:00
#SBATCH -A ASC25081

module load cuda/12.4

set -e
cd $SCRATCH
eval "$(conda shell.bash hook)"
conda activate bb_audit_dpsgd
cd bb-audit-dpsgd

# Common hyperparameters for CIFAR-10
DATA_NAME="cifar10"
MODEL_NAME="cnn"
N_REPS=400
N_EPOCHS=100
LR=3
MAX_GRAD_NORM=1
EPSILON=10.0
DELTA=1e-5
BATCH_SIZE=3125
BLOCK_SIZE=3125
SEED=0
CANARY_PT="clipbkd_canary.pt"
DEFENSE_K=5

# Parent output folder
PARENT_OUT="cifar10_clipbkd_canary_defense"
mkdir -p ${PARENT_OUT}

echo "=========================================="
echo "Running CIFAR-10 ClipBKD Canary Audit (With Defense)"
echo "Dataset: ${DATA_NAME}"
echo "Model: ${MODEL_NAME}"
echo "Canary: ${CANARY_PT}"
echo "Epsilon: ${EPSILON}"
echo "Defense K: ${DEFENSE_K}"
echo "N_reps: ${N_REPS}"
echo "=========================================="

# Run the audit
srun --ntasks=$SLURM_NTASKS --nodes=$SLURM_JOB_NUM_NODES bash -c '
  MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1);
  MASTER_PORT=29500;
  RANK=$SLURM_PROCID;
  torchrun --nnodes=$SLURM_JOB_NUM_NODES --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    parallel_audit_model.py \
    --data_name '"${DATA_NAME}"' \
    --model_name '"${MODEL_NAME}"' \
    --canary_pt '"${CANARY_PT}"' \
    --n_reps '"${N_REPS}"' \
    --n_epochs '"${N_EPOCHS}"' \
    --lr '"${LR}"' \
    --max_grad_norm '"${MAX_GRAD_NORM}"' \
    --epsilon '"${EPSILON}"' \
    --delta '"${DELTA}"' \
    --batch_size '"${BATCH_SIZE}"' \
    --block_size '"${BLOCK_SIZE}"' \
    --seed '"${SEED}"' \
    --fixed_init \
    --aug_mult 1 \
    --holdout_audit \
    --defense \
    --defense_k '"${DEFENSE_K}"' \
    --out '"${PARENT_OUT}"'
'

echo "=========================================="
echo "CIFAR-10 ClipBKD Canary Audit (With Defense) completed!"
echo "Results saved in: ${PARENT_OUT}/"
echo "=========================================="
