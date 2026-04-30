#!/bin/bash
#SBATCH -J gradient_cancel_mnist
#SBATCH -o status/gradient_cancel_mnist_%A_%a.o
#SBATCH -e status/gradient_cancel_mnist_%A_%a.e
#SBATCH -p gh
#SBATCH -N 5
#SBATCH --ntasks-per-node=1
#SBATCH -t 7:00:00
#SBATCH -A ASC25081
#SBATCH --array=0-1

module load cuda/12.4

set -e
cd $SCRATCH
eval "$(conda shell.bash hook)"
conda activate bb_audit_dpsgd
cd bb-audit-dpsgd

mkdir -p status

DATA_NAME="mnist"
MODEL_NAME="cnn"
N_REPS=400
N_EPOCHS=100
LR=3
MAX_GRAD_NORM=1.0
EPSILON=10.0
DELTA=1e-5
BATCH_SIZE=4000
BLOCK_SIZE=4000
SEED=0
DEFENSE_K=5
DEFENSE_SCORE_FN="grad_norm_unclipped"
DEFENSE_SCORE_NORM="linf"
CANARY_PT="grad_cancel_test/gradient_space_canaries.pt"
PARENT_OUT="grad_cancel_test"

# Array: 0 = no defense, 1 = defense
case ${SLURM_ARRAY_TASK_ID} in
    0)
        DEFENSE_ARGS=""
        OUT_NAME="audit_no_defense"
        ;;
    1)
        DEFENSE_ARGS="--defense --defense_k ${DEFENSE_K} --defense_score_fn ${DEFENSE_SCORE_FN} --defense_score_norm ${DEFENSE_SCORE_NORM}"
        OUT_NAME="audit_defense"
        ;;
esac

echo "=========================================="
echo "Task ${SLURM_ARRAY_TASK_ID}: ${OUT_NAME}"
echo "Canary: ${CANARY_PT}"
echo "N_REPS: ${N_REPS}  N_EPOCHS: ${N_EPOCHS}  Epsilon: ${EPSILON}"
echo "Defense args: ${DEFENSE_ARGS}"
echo "=========================================="

srun --ntasks=$SLURM_NTASKS --nodes=$SLURM_JOB_NUM_NODES bash -c '
  MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1);
  MASTER_PORT=$((29500 + '"${SLURM_ARRAY_TASK_ID}"'));
  torchrun --nnodes=$SLURM_JOB_NUM_NODES --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    parallel_audit_multi_canary.py \
    --data_name '"${DATA_NAME}"' \
    --model_name '"${MODEL_NAME}"' \
    --n_reps '"${N_REPS}"' \
    --n_epochs '"${N_EPOCHS}"' \
    --lr '"${LR}"' \
    --batch_size '"${BATCH_SIZE}"' \
    --block_size '"${BLOCK_SIZE}"' \
    --epsilon '"${EPSILON}"' \
    --delta '"${DELTA}"' \
    --max_grad_norm '"${MAX_GRAD_NORM}"' \
    --sampling poisson \
    --holdout_audit \
    --gradient_space_canary_pt '"${CANARY_PT}"' \
    --target_type gradient_space_canary \
    --gradient_space_score_fn hot_param \
    --seed '"${SEED}"' \
    --fixed_init \
    '"${DEFENSE_ARGS}"' \
    --out '"${PARENT_OUT}"'/'"${OUT_NAME}"''

echo "Task ${SLURM_ARRAY_TASK_ID} done: ${OUT_NAME}"
