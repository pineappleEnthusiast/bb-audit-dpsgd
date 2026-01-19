#!/bin/bash
#SBATCH -J mnist_1000_reps
#SBATCH -o status/mnist_1000_reps_%A_%a.o
#SBATCH -e status/mnist_1000_reps_%A_%a.e
#SBATCH -p gh
#SBATCH -N 20
#SBATCH --ntasks-per-node=1
#SBATCH -t 4:00:00
#SBATCH -A ASC25081
#SBATCH --array=0-1

module load cuda/12.4

set -e
cd $SCRATCH
eval "$(conda shell.bash hook)"
conda activate bb_audit_dpsgd
cd bb-audit-dpsgd

# Common hyperparameters for MNIST
DATA_NAME="mnist"
MODEL_NAME="cnn"
N_REPS=1000
N_EPOCHS=100
LR=3
MAX_GRAD_NORM=1
EPSILON=10.0
DELTA=1e-5
BATCH_SIZE=4000
BLOCK_SIZE=4000
SEED=0
TARGET_TYPE="blank"
DEFENSE_K=5
DEFENSE_SCORE_FN="grad_norm"
DEFENSE_SCORE_NORM="linf"

# Parent output folder
PARENT_OUT="mnist_1000_reps"
mkdir -p ${PARENT_OUT}

# Job array mapping:
# 0: mnist_no_defense
# 1: mnist_defense

case ${SLURM_ARRAY_TASK_ID} in
    0)
        DEFENSE_FLAG=""
        OUT_NAME="mnist_no_defense"
        ;;
    1)
        DEFENSE_FLAG="--defense --defense_k ${DEFENSE_K} --defense_score_fn ${DEFENSE_SCORE_FN} --defense_score_norm ${DEFENSE_SCORE_NORM} --defense_apply_ascent"
        OUT_NAME="mnist_defense"
        ;;
esac

echo "=========================================="
echo "Job Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Dataset: ${DATA_NAME}"
echo "Model: ${MODEL_NAME}"
echo "N_REPS: ${N_REPS}"
echo "Epsilon: ${EPSILON}"
echo "Output: ${OUT_NAME}"
echo "=========================================="

# Run the audit
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
    ${DEFENSE_FLAG} \
    --out ${PARENT_OUT}/${OUT_NAME}
"
echo "=========================================="
echo "Task ${SLURM_ARRAY_TASK_ID} completed: ${OUT_NAME}"
echo "Results saved in: ${PARENT_OUT}/${OUT_NAME}/"
echo "=========================================="
