#!/bin/bash
#SBATCH -J grad_ascent_ablation
#SBATCH -o status/grad_ascent_ablation_%A_%a.o
#SBATCH -e status/grad_ascent_ablation_%A_%a.e
#SBATCH -p gh
#SBATCH -N 20
#SBATCH --ntasks-per-node=1
#SBATCH -t 4:00:00
#SBATCH -A ASC25081
#SBATCH --array=0-8

module load cuda/12.4

set -e
cd $SCRATCH
eval "$(conda shell.bash hook)"
conda activate bb_audit_dpsgd
cd bb-audit-dpsgd

# Common hyperparameters
N_REPS=400
N_EPOCHS=100
MAX_GRAD_NORM=1
DELTA=1e-5
SEED=0
TARGET_TYPE="blank"
DEFENSE_K=5
DEFENSE_SCORE_FN="grad_norm"
DEFENSE_SCORE_NORM="linf"

# Parent output folder
PARENT_OUT="gradient_ascent_ablation"
mkdir -p ${PARENT_OUT}

# Define dataset configurations
# MNIST: eps=10, lr=3, batch=4000
# CIFAR10: eps=10, lr=3, batch=3125
# Purchase: eps=10, lr=0.5, batch=12143

# Job array mapping:
# 0: mnist_no_defense
# 1: mnist_defense_no_ascent
# 2: mnist_defense_with_ascent
# 3: cifar10_no_defense
# 4: cifar10_defense_no_ascent
# 5: cifar10_defense_with_ascent
# 6: purchase_no_defense
# 7: purchase_defense_no_ascent
# 8: purchase_defense_with_ascent

case ${SLURM_ARRAY_TASK_ID} in
    0)
        DATA_NAME="mnist"
        MODEL_NAME="cnn"
        EPSILON=10.0
        LR=3
        BATCH_SIZE=4000
        BLOCK_SIZE=4000
        DEFENSE_FLAG=""
        ASCENT_FLAG=""
        OUT_NAME="mnist_no_defense"
        ;;
    1)
        DATA_NAME="mnist"
        MODEL_NAME="cnn"
        EPSILON=10.0
        LR=3
        BATCH_SIZE=4000
        BLOCK_SIZE=4000
        DEFENSE_FLAG="--defense --defense_k ${DEFENSE_K} --defense_score_fn ${DEFENSE_SCORE_FN} --defense_score_norm ${DEFENSE_SCORE_NORM}"
        ASCENT_FLAG=""
        OUT_NAME="mnist_defense_no_ascent"
        ;;
    2)
        DATA_NAME="mnist"
        MODEL_NAME="cnn"
        EPSILON=10.0
        LR=3
        BATCH_SIZE=4000
        BLOCK_SIZE=4000
        DEFENSE_FLAG="--defense --defense_k ${DEFENSE_K} --defense_score_fn ${DEFENSE_SCORE_FN} --defense_score_norm ${DEFENSE_SCORE_NORM}"
        ASCENT_FLAG="--defense_apply_ascent"
        OUT_NAME="mnist_defense_with_ascent"
        ;;
    3)
        DATA_NAME="cifar10"
        MODEL_NAME="cnn"
        EPSILON=10.0
        LR=3
        BATCH_SIZE=3125
        BLOCK_SIZE=3125
        DEFENSE_FLAG=""
        ASCENT_FLAG=""
        OUT_NAME="cifar10_no_defense"
        ;;
    4)
        DATA_NAME="cifar10"
        MODEL_NAME="cnn"
        EPSILON=10.0
        LR=3
        BATCH_SIZE=3125
        BLOCK_SIZE=3125
        DEFENSE_FLAG="--defense --defense_k ${DEFENSE_K} --defense_score_fn ${DEFENSE_SCORE_FN} --defense_score_norm ${DEFENSE_SCORE_NORM}"
        ASCENT_FLAG=""
        OUT_NAME="cifar10_defense_no_ascent"
        ;;
    5)
        DATA_NAME="cifar10"
        MODEL_NAME="cnn"
        EPSILON=10.0
        LR=3
        BATCH_SIZE=3125
        BLOCK_SIZE=3125
        DEFENSE_FLAG="--defense --defense_k ${DEFENSE_K} --defense_score_fn ${DEFENSE_SCORE_FN} --defense_score_norm ${DEFENSE_SCORE_NORM}"
        ASCENT_FLAG="--defense_apply_ascent"
        OUT_NAME="cifar10_defense_with_ascent"
        ;;
    6)
        DATA_NAME="purchase"
        MODEL_NAME="mlp"
        EPSILON=10.0
        LR=10
        BATCH_SIZE=12143
        BLOCK_SIZE=12143
        DEFENSE_FLAG=""
        ASCENT_FLAG=""
        OUT_NAME="purchase_no_defense"
        ;;
    7)
        DATA_NAME="purchase"
        MODEL_NAME="mlp"
        EPSILON=10.0
        LR=10
        BATCH_SIZE=12143
        BLOCK_SIZE=12143
        DEFENSE_FLAG="--defense --defense_k ${DEFENSE_K} --defense_score_fn ${DEFENSE_SCORE_FN} --defense_score_norm ${DEFENSE_SCORE_NORM}"
        ASCENT_FLAG=""
        OUT_NAME="purchase_defense_no_ascent"
        ;;
    8)
        DATA_NAME="purchase"
        MODEL_NAME="mlp"
        EPSILON=10.0
        LR=10
        BATCH_SIZE=12143
        BLOCK_SIZE=12143
        DEFENSE_FLAG="--defense --defense_k ${DEFENSE_K} --defense_score_fn ${DEFENSE_SCORE_FN} --defense_score_norm ${DEFENSE_SCORE_NORM}"
        ASCENT_FLAG="--defense_apply_ascent"
        OUT_NAME="purchase_defense_with_ascent"
        ;;
esac

echo "=========================================="
echo "Job Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Dataset: ${DATA_NAME}"
echo "Model: ${MODEL_NAME}"
echo "Epsilon: ${EPSILON}"
echo "Output: ${OUT_NAME}"
echo "=========================================="

# Run the audit
srun --ntasks=$SLURM_NTASKS --nodes=$SLURM_JOB_NUM_NODES bash -c '
  MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1);
  MASTER_PORT=$((29500 + '"${SLURM_ARRAY_TASK_ID}"'));
  RANK=$SLURM_PROCID;
  torchrun --nnodes=$SLURM_JOB_NUM_NODES --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    parallel_audit_model.py \
    --data_name '"${DATA_NAME}"' \
    --model_name '"${MODEL_NAME}"' \
    --n_reps '"${N_REPS}"' \
    --n_epochs '"${N_EPOCHS}"' \
    --lr '"${LR}"' \
    --max_grad_norm '"${MAX_GRAD_NORM}"' \
    --epsilon '"${EPSILON}"' \
    --delta '"${DELTA}"' \
    --target_type '"${TARGET_TYPE}"' \
    --batch_size '"${BATCH_SIZE}"' \
    --block_size '"${BLOCK_SIZE}"' \
    --seed '"${SEED}"' \
    --fixed_init \
    --aug_mult 1 \
    --holdout_audit \
    '"${DEFENSE_FLAG}"' \
    '"${ASCENT_FLAG}"' \
    --out '"${PARENT_OUT}/${OUT_NAME}"'

echo "=========================================="
echo "Task ${SLURM_ARRAY_TASK_ID} completed: ${OUT_NAME}"
echo "Results saved in: ${PARENT_OUT}/${OUT_NAME}/"
echo "=========================================="
