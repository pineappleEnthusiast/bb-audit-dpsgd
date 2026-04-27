#!/bin/bash
#SBATCH -J ablation_score_fn
#SBATCH -o status/ablation_score_fn_%A_%a.o
#SBATCH -e status/ablation_score_fn_%A_%a.e
#SBATCH -p gh
#SBATCH -N 5
#SBATCH --ntasks-per-node=1
#SBATCH -t 5:00:00
#SBATCH -A ASC25081
#SBATCH --array=0-7

module load cuda/12.4

set -e
cd $WORK
eval "$(conda shell.bash hook)"
conda activate bb_audit_dpsgd
cd bb-audit-dpsgd

mkdir -p status

# -----------------------------------------------------------------------
# Fixed hyperparameters
# -----------------------------------------------------------------------
DATA_NAME="cifar10"
MODEL_NAME="cnn"
LR=3
BATCH_SIZE=3125
BLOCK_SIZE=3125
AUG_MULT=1
N_REPS=400
N_EPOCHS=100
MAX_GRAD_NORM=1
EPSILON=10.0
DELTA=1e-5
SEED=0
TARGET_TYPE="blank"
DEFENSE_K=5
PARENT_OUT="ablation_score_fn"

mkdir -p ${PARENT_OUT}

# -----------------------------------------------------------------------
# Experiment configurations
# Format: "score_fn|score_norm|output_name"
#
# Scoring functions tested:
#   0  cos_theta0          – cosine similarity to initial params
#   1  grad_norm (L2)      – per-sample gradient L2 norm
#   2  grad_norm (L-inf)   – per-sample gradient L-inf norm
#   3  gradient_kurtosis   – kurtosis of gradient distribution
#   4  norm_x_dir_unique.  – norm × direction uniqueness
#   5  pred_entropy        – prediction entropy
#   6  prediction_margin   – prediction margin (confidence gap)
#   7  rand_proj_var       – variance of random projections
#   8  no defense baseline – same setup with defense disabled
# -----------------------------------------------------------------------
experiments=(
    "cos_theta0|linf|cifar10_cnn_blank_eps10_cos_theta0"
    "grad_norm|l2|cifar10_cnn_blank_eps10_grad_norm_l2"
    "grad_norm|linf|cifar10_cnn_blank_eps10_grad_norm_linf"
    "gradient_kurtosis|linf|cifar10_cnn_blank_eps10_gradient_kurtosis"
    "norm_x_dir_uniqueness|linf|cifar10_cnn_blank_eps10_norm_x_dir_uniqueness"
    "pred_entropy|linf|cifar10_cnn_blank_eps10_pred_entropy"
    "prediction_margin|linf|cifar10_cnn_blank_eps10_prediction_margin"
    "rand_proj_var|linf|cifar10_cnn_blank_eps10_rand_proj_var"
)

IFS='|' read -r SCORE_FN SCORE_NORM OUT_NAME <<< "${experiments[$SLURM_ARRAY_TASK_ID]}"

echo "=========================================="
echo "Job Array Task ID : ${SLURM_ARRAY_TASK_ID}"
echo "Dataset           : ${DATA_NAME}"
echo "Model             : ${MODEL_NAME}"
echo "Epsilon           : ${EPSILON}"
echo "Score function    : ${SCORE_FN}"
echo "Score norm        : ${SCORE_NORM}"
echo "Output name       : ${OUT_NAME}"
echo "=========================================="

# Build defense flags conditionally
if [ "${SCORE_FN}" = "none" ]; then
    DEFENSE_ARGS=""
else
    DEFENSE_ARGS="--defense --defense_k ${DEFENSE_K} --defense_score_fn ${SCORE_FN} --defense_score_norm ${SCORE_NORM}"
fi

srun --ntasks=$SLURM_NTASKS --nodes=$SLURM_JOB_NUM_NODES bash -c '
  MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1);
  MASTER_PORT=$((29500 + '"${SLURM_ARRAY_TASK_ID}"'));
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
    '"${DEFENSE_ARGS}"' \
    --out '"${PARENT_OUT}/${OUT_NAME}"''

echo "=========================================="
echo "Task ${SLURM_ARRAY_TASK_ID} completed: ${OUT_NAME}"
echo "Results saved in: ${PARENT_OUT}/${OUT_NAME}/"
echo "=========================================="
