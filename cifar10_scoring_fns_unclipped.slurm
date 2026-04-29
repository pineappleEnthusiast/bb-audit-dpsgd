#!/bin/bash
#SBATCH -J cifar10_scoring_fns_unclipped
#SBATCH -o status/cifar10_scoring_fns_unclipped_%A_%a.o
#SBATCH -e status/cifar10_scoring_fns_unclipped_%A_%a.e
#SBATCH -p gh
#SBATCH -N 5
#SBATCH --ntasks-per-node=1
#SBATCH -t 5:00:00
#SBATCH -A ASC25081
#SBATCH --array=0-9

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
TARGET_TYPE="blank"
DEFENSE_K=5

# Parent output folder
PARENT_OUT="cifar10_scoring_functions_unclipped"
mkdir -p ${PARENT_OUT}

# Same 8 scoring functions as cifar10_scoring_fns.slurm, with _unclipped appended
# to all gradient-based ones. prediction_margin and pred_entropy are logit-based
# so they are kept as-is. Two clipped grad_norm variants added as benchmarks.
#
# Array format: "score_fn|score_norm|output_name"
experiments=(
    "grad_norm_unclipped|linf|cifar10_grad_norm_unclipped_linf"
    "grad_norm_unclipped|l2|cifar10_grad_norm_unclipped_l2"
    "cos_theta0_unclipped|l2|cifar10_cos_theta0_unclipped"
    "gradient_kurtosis_unclipped|l2|cifar10_gradient_kurtosis_unclipped"
    "norm_x_dir_uniqueness_unclipped|linf|cifar10_norm_x_dir_uniqueness_unclipped"
    "prediction_margin|l2|cifar10_prediction_margin_unclipped"
    "rand_proj_var_unclipped|l2|cifar10_rand_proj_var_unclipped"
    "pred_entropy|l2|cifar10_pred_entropy_unclipped"
    "grad_norm|linf|cifar10_grad_norm_linf"
    "grad_norm|l2|cifar10_grad_norm_l2"
)

# Get current experiment configuration
IFS='|' read -r SCORE_FN SCORE_NORM OUT_NAME <<< "${experiments[$SLURM_ARRAY_TASK_ID]}"

echo "=========================================="
echo "Job Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Dataset: ${DATA_NAME}"
echo "Model: ${MODEL_NAME}"
echo "Epsilon: ${EPSILON}"
echo "Scoring Function: ${SCORE_FN}"
echo "Scoring Norm: ${SCORE_NORM}"
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
    --defense \
    --defense_k '"${DEFENSE_K}"' \
    --defense_score_fn '"${SCORE_FN}"' \
    --defense_score_norm '"${SCORE_NORM}"' \
    --holdout_audit \
    --out '"${PARENT_OUT}"'/'"${OUT_NAME}"''

echo "=========================================="
echo "Task ${SLURM_ARRAY_TASK_ID} completed: ${OUT_NAME}"
echo "Results saved in: ${PARENT_OUT}/${OUT_NAME}/"
echo "=========================================="
