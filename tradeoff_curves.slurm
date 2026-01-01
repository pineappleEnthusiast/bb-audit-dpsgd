#!/bin/bash
#SBATCH -J tradeoff_curves
#SBATCH -o status/tradeoff_curves_%A_%a.o
#SBATCH -e status/tradeoff_curves_%A_%a.e
#SBATCH -p gh
#SBATCH -N 20
#SBATCH --ntasks-per-node=1
#SBATCH -t 4:00:00
#SBATCH -A ASC25081
#SBATCH --array=0-29

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
DEFENSE_SCORE_FN="grad_norm"
DEFENSE_SCORE_NORM="l2"
DEFENSE_K=5

# Parent output folder
PARENT_OUT="tradeoff_curves"
mkdir -p ${PARENT_OUT}

# Define experiment configurations
# Array format: "dataset|model|lr|batch_size|block_size|epsilon|aug_mult|defense|time|output_name"
experiments=(
    # MNIST experiments (eps 6, 8, 10 × defense/no_defense)
    "mnist|cnn|3|4000|4000|6.0|1|false|4:00:00|mnist_no_defense_eps6"
    "mnist|cnn|3|4000|4000|6.0|1|true|4:00:00|mnist_defense_eps6"
    "mnist|cnn|3|4000|4000|8.0|1|false|4:00:00|mnist_no_defense_eps8"
    "mnist|cnn|3|4000|4000|8.0|1|true|4:00:00|mnist_defense_eps8"
    "mnist|cnn|3|4000|4000|10.0|1|false|4:00:00|mnist_no_defense_eps10"
    "mnist|cnn|3|4000|4000|10.0|1|true|4:00:00|mnist_defense_eps10"
    
    # Purchase experiments (eps 6, 8, 10 × defense/no_defense)
    "purchase|mlp|10|12143|12143|6.0|1|false|4:00:00|purchase_no_defense_eps6"
    "purchase|mlp|10|12143|12143|6.0|1|true|4:00:00|purchase_defense_eps6"
    "purchase|mlp|10|12143|12143|8.0|1|false|4:00:00|purchase_no_defense_eps8"
    "purchase|mlp|10|12143|12143|8.0|1|true|4:00:00|purchase_defense_eps8"
    "purchase|mlp|10|12143|12143|10.0|1|false|4:00:00|purchase_no_defense_eps10"
    "purchase|mlp|10|12143|12143|10.0|1|true|4:00:00|purchase_defense_eps10"
    
    # CIFAR-10 CNN experiments (eps 6, 8, 10 × defense/no_defense)
    "cifar10|cnn|3|3125|3125|6.0|1|false|4:00:00|cifar10_no_defense_eps6"
    "cifar10|cnn|3|3125|3125|6.0|1|true|4:00:00|cifar10_defense_eps6"
    "cifar10|cnn|3|3125|3125|8.0|1|false|4:00:00|cifar10_no_defense_eps8"
    "cifar10|cnn|3|3125|3125|8.0|1|true|4:00:00|cifar10_defense_eps8"
    "cifar10|cnn|3|3125|3125|10.0|1|false|4:00:00|cifar10_no_defense_eps10"
    "cifar10|cnn|3|3125|3125|10.0|1|true|4:00:00|cifar10_defense_eps10"
    
    # CIFAR-10 CNN + augmult4 experiments (eps 6, 8, 10 × defense/no_defense)
    "cifar10|cnn|3|3125|2000|6.0|4|false|15:00:00|cifar10_augmult4_no_defense_eps6"
    "cifar10|cnn|3|3125|2000|6.0|4|true|15:00:00|cifar10_augmult4_defense_eps6"
    "cifar10|cnn|3|3125|2000|8.0|4|false|15:00:00|cifar10_augmult4_no_defense_eps8"
    "cifar10|cnn|3|3125|2000|8.0|4|true|15:00:00|cifar10_augmult4_defense_eps8"
    "cifar10|cnn|3|3125|2000|10.0|4|false|15:00:00|cifar10_augmult4_no_defense_eps10"
    "cifar10|cnn|3|3125|2000|10.0|4|true|15:00:00|cifar10_augmult4_defense_eps10"
    
    # CIFAR-10 WideResNet experiments (eps 6, 8, 10 × defense/no_defense)
    "cifar10|wideresnet|3|3125|1000|6.0|1|false|15:00:00|cifar10_wrn_no_defense_eps6"
    "cifar10|wideresnet|3|3125|1000|6.0|1|true|15:00:00|cifar10_wrn_defense_eps6"
    "cifar10|wideresnet|3|3125|1000|8.0|1|false|15:00:00|cifar10_wrn_no_defense_eps8"
    "cifar10|wideresnet|3|3125|1000|8.0|1|true|15:00:00|cifar10_wrn_defense_eps8"
    "cifar10|wideresnet|3|3125|1000|10.0|1|false|15:00:00|cifar10_wrn_no_defense_eps10"
    "cifar10|wideresnet|3|3125|1000|10.0|1|true|15:00:00|cifar10_wrn_defense_eps10"
)

# Get current experiment configuration
IFS='|' read -r DATA_NAME MODEL_NAME LR BATCH_SIZE BLOCK_SIZE EPSILON AUG_MULT DEFENSE TIME_LIMIT OUT_NAME <<< "${experiments[$SLURM_ARRAY_TASK_ID]}"

# Update time limit for this job (note: this won't change the already-submitted job, but documents it)
echo "=========================================="
echo "Job Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Dataset: ${DATA_NAME}"
echo "Model: ${MODEL_NAME}"
echo "Epsilon: ${EPSILON}"
echo "Learning Rate: ${LR}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Aug Mult: ${AUG_MULT}"
echo "Defense: ${DEFENSE}"
echo "Time Limit: ${TIME_LIMIT}"
echo "Output: ${OUT_NAME}"
echo "=========================================="

# Build defense arguments
DEFENSE_ARGS=""
if [ "${DEFENSE}" = "true" ]; then
    DEFENSE_ARGS="--defense --defense_k ${DEFENSE_K} --defense_score_fn ${DEFENSE_SCORE_FN} --defense_score_norm ${DEFENSE_SCORE_NORM} --defense_apply_ascent"
fi

# Handle purchase dataset special case
N_DF_ARG=""
if [ "${DATA_NAME}" = "purchase" ]; then
    N_DF_ARG="--n_df 153800"
fi

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
    '"${N_DF_ARG}"' \
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
    --aug_mult '"${AUG_MULT}"' \
    '"${DEFENSE_ARGS}"' \
    --holdout_audit \
    --out '"${PARENT_OUT}"'/'"${OUT_NAME}"''

echo "=========================================="
echo "Task ${SLURM_ARRAY_TASK_ID} completed: ${OUT_NAME}"
echo "Results saved in: ${PARENT_OUT}/${OUT_NAME}/"
echo "=========================================="
