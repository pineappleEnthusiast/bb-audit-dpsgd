#!/bin/bash
#SBATCH -J ablation_blank_alpha
#SBATCH -o status/ablation_blank_alpha_%A_%a.o
#SBATCH -e status/ablation_blank_alpha_%A_%a.e
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

mkdir -p status

# Common hyperparameters
N_REPS=400
N_EPOCHS=100
MAX_GRAD_NORM=1
EPSILON=10.0
DELTA=1e-5
SEED=0
TARGET_TYPE="mislabeled_alpha"
MISLABELED_TARGET_CLASS=1     # wrong label always assigned to the canary
DEFENSE_SCORE_FN="grad_norm_unclipped"
DEFENSE_SCORE_NORM="linf"

# Parent output folder
PARENT_OUT="ablation_blank_alpha"
mkdir -p ${PARENT_OUT}

# Define experiment configurations
# Array format: "dataset|model|lr|batch_size|block_size|defense_k|alpha|output_name"
experiments=(
    "mnist|cnn|3|4000|4000|5|0.0|mnist_alpha0.0"
    "mnist|cnn|3|4000|4000|5|0.25|mnist_alpha0.25"
    "mnist|cnn|3|4000|4000|5|0.5|mnist_alpha0.5"
    "mnist|cnn|3|4000|4000|5|0.75|mnist_alpha0.75"
    "mnist|cnn|3|4000|4000|5|1.0|mnist_alpha1.0"
    "cifar10|cnn|3|3125|3125|5|0.0|cifar10_alpha0.0"
    "cifar10|cnn|3|3125|3125|5|0.25|cifar10_alpha0.25"
    "cifar10|cnn|3|3125|3125|5|0.5|cifar10_alpha0.5"
    "cifar10|cnn|3|3125|3125|5|0.75|cifar10_alpha0.75"
    "cifar10|cnn|3|3125|3125|5|1.0|cifar10_alpha1.0"
)

# Get current experiment configuration
IFS='|' read -r DATA_NAME MODEL_NAME LR BATCH_SIZE BLOCK_SIZE DEFENSE_K BLANK_ALPHA OUT_NAME <<< "${experiments[$SLURM_ARRAY_TASK_ID]}"

echo "=========================================="
echo "Job Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Dataset: ${DATA_NAME}"
echo "Model: ${MODEL_NAME}"
echo "Epsilon: ${EPSILON}"
echo "Learning Rate: ${LR}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Defense K: ${DEFENSE_K}"
echo "Blank Alpha: ${BLANK_ALPHA}"
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
    --blank_alpha '"${BLANK_ALPHA}"' \
    --mislabeled_target_class '"${MISLABELED_TARGET_CLASS}"' \
    --batch_size '"${BATCH_SIZE}"' \
    --block_size '"${BLOCK_SIZE}"' \
    --seed '"${SEED}"' \
    --fixed_init \
    --aug_mult 1 \
    --sampling poisson \
    --defense \
    --defense_k '"${DEFENSE_K}"' \
    --defense_score_fn '"${DEFENSE_SCORE_FN}"' \
    --defense_score_norm '"${DEFENSE_SCORE_NORM}"' \
    --holdout_audit \
    --out '"${PARENT_OUT}/${OUT_NAME}"'
'
echo "=========================================="
echo "Task ${SLURM_ARRAY_TASK_ID} completed: ${OUT_NAME}"
echo "Results saved in: ${PARENT_OUT}/${OUT_NAME}/"
echo "=========================================="
