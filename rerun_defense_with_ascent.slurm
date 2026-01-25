#!/bin/bash
#SBATCH -J rerun_defense_ascent
#SBATCH -o status/rerun_defense_ascent_%A_%a.o
#SBATCH -e status/rerun_defense_ascent_%A_%a.e
#SBATCH -p gh
#SBATCH -N 10
#SBATCH --ntasks-per-node=1
#SBATCH -t 8:00:00
#SBATCH -A ASC25081
#SBATCH --array=0-2

module load cuda/12.4

set -e
cd $SCRATCH
source ~/miniconda3/etc/profile.d/conda.sh
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
PARENT_OUT="redo_gradient_ascent_ablation"
mkdir -p ${PARENT_OUT}

# Job array mapping (only defense with ascent):
# 0: mnist_defense_with_ascent
# 1: cifar10_defense_with_ascent
# 2: purchase_defense_with_ascent

case ${SLURM_ARRAY_TASK_ID} in
    0)
        DATA_NAME="mnist"
        MODEL_NAME="cnn"
        EPSILON=10.0
        LR=3
        BATCH_SIZE=4000
        BLOCK_SIZE=4000
        OUT_NAME="mnist_defense_with_ascent"
        ;;
    1)
        DATA_NAME="cifar10"
        MODEL_NAME="cnn"
        EPSILON=10.0
        LR=3
        BATCH_SIZE=3125
        BLOCK_SIZE=3125
        OUT_NAME="cifar10_defense_with_ascent"
        ;;
    2)
        DATA_NAME="purchase"
        MODEL_NAME="mlp"
        EPSILON=10.0
        LR=10
        BATCH_SIZE=12143
        BLOCK_SIZE=12143
        N_DF=153800
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

# Handle purchase dataset special case
N_DF_ARG=""
if [ "${DATA_NAME}" = "purchase" ]; then
    N_DF_ARG="--n_df ${N_DF}"
fi

# Run the audit with defense and ascent enabled
srun --ntasks=$SLURM_NTASKS --nodes=$SLURM_JOB_NUM_NODES bash -c "
  MASTER_ADDR=\$(scontrol show hostnames \$SLURM_JOB_NODELIST | head -n 1);
  MASTER_PORT=\$((29500 + ${SLURM_ARRAY_TASK_ID}));
  RANK=\$SLURM_PROCID;
  torchrun --nnodes=\$SLURM_JOB_NUM_NODES --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=\${MASTER_ADDR}:\${MASTER_PORT} \
    parallel_audit_model.py \
    --data_name ${DATA_NAME} \
    --model_name ${MODEL_NAME} \
    ${N_DF_ARG} \
    --n_reps ${N_REPS} \
    --n_epochs ${N_EPOCHS} \
    --lr ${LR} \
    --max_grad_norm ${MAX_GRAD_NORM} \
    --epsilon ${EPSILON} \
    --delta ${DELTA} \
    --target_type ${TARGET_TYPE} \
    --batch_size ${BATCH_SIZE} \
    --block_size ${BLOCK_SIZE} \
    --seed ${SEED} \
    --fixed_init \
    --aug_mult 1 \
    --holdout_audit \
    --defense \
    --defense_k ${DEFENSE_K} \
    --defense_score_fn ${DEFENSE_SCORE_FN} \
    --defense_score_norm ${DEFENSE_SCORE_NORM} \
    --defense_apply_ascent \
    --out ${PARENT_OUT}/${OUT_NAME}
"
echo "=========================================="
echo "Task ${SLURM_ARRAY_TASK_ID} completed: ${OUT_NAME}"
echo "Results saved in: ${PARENT_OUT}/${OUT_NAME}/"
echo "=========================================="
