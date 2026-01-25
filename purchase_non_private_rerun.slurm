#!/bin/bash
#SBATCH -J purchase_non_private
#SBATCH -o purchase_non_private.o%j
#SBATCH -e purchase_non_private.e%j
#SBATCH -p gh
#SBATCH -N 5
#SBATCH -n 5
#SBATCH --ntasks-per-node=1
#SBATCH -t 20:00:00
#SBATCH -A ASC25081

module load cuda/12.4

set -e
cd $SCRATCH
eval "$(conda shell.bash hook)"
conda activate bb_audit_dpsgd
cd bb-audit-dpsgd

# Common hyperparameters
N_REPS=400
N_EPOCHS=100
DELTA=1e-5
SEED=0
TARGET_TYPE="blank"
DEFENSE_SCORE_FN="cos_theta0"
DEFENSE_SCORE_NORM="linf"
DEFENSE_K=5

# Purchase-specific parameters
DATA_NAME="purchase"
MODEL_NAME="mlp"
LR=0.5
BATCH_SIZE=12143
BLOCK_SIZE=12143
AUG_MULT=1
N_DF=153800

# Parent output folder
PARENT_OUT="non_private_audits_rerun"
mkdir -p ${PARENT_OUT}

echo "=========================================="
echo "Step 1: Running audit without defense"
echo "=========================================="

# Step 1: Run audit without defense
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
    --n_df '"${N_DF}"' \
    --n_reps '"${N_REPS}"' \
    --n_epochs '"${N_EPOCHS}"' \
    --lr '"${LR}"' \
    --epsilon -1 \
    --max_grad_norm -1 \
    --target_type '"${TARGET_TYPE}"' \
    --batch_size '"${BATCH_SIZE}"' \
    --block_size '"${BLOCK_SIZE}"' \
    --seed '"${SEED}"' \
    --fixed_init \
    --aug_mult '"${AUG_MULT}"' \
    --holdout_audit \
    --out '"${PARENT_OUT}/purchase_no_defense_non_private"'
'

echo "=========================================="
echo "Step 2: Running audit with defense"
echo "=========================================="

# Step 2: Run audit with defense
srun --ntasks=$SLURM_NTASKS --nodes=$SLURM_JOB_NUM_NODES bash -c '
  MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1);
  MASTER_PORT=29501;
  RANK=$SLURM_PROCID;
  torchrun --nnodes=$SLURM_JOB_NUM_NODES --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    parallel_audit_model.py \
    --data_name '"${DATA_NAME}"' \
    --model_name '"${MODEL_NAME}"' \
    --n_df '"${N_DF}"' \
    --n_reps '"${N_REPS}"' \
    --n_epochs '"${N_EPOCHS}"' \
    --lr '"${LR}"' \
    --epsilon -1 \
    --max_grad_norm -1 \
    --target_type '"${TARGET_TYPE}"' \
    --batch_size '"${BATCH_SIZE}"' \
    --block_size '"${BLOCK_SIZE}"' \
    --seed '"${SEED}"' \
    --fixed_init \
    --aug_mult '"${AUG_MULT}"' \
    --defense \
    --defense_k '"${DEFENSE_K}"' \
    --defense_score_fn '"${DEFENSE_SCORE_FN}"' \
    --defense_score_norm '"${DEFENSE_SCORE_NORM}"' \
    --defense_apply_ascent \
    --holdout_audit \
    --out '"${PARENT_OUT}/purchase_defense_non_private"'
'

echo "=========================================="
echo "All steps completed successfully"
echo "=========================================="
