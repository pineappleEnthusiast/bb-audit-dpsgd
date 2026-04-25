#!/bin/bash
set -e

OUTPUT_DIR="majority_canary_colored_mnist_eps10"
EPSILON=10
DELTA=1e-5
MAX_GRAD_NORM=1.0
N_REPS=200
N_NODES=5
MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
MASTER_PORT=29500

# Majority-canary audit on Colored MNIST (75% majority split), private regime (eps=10).
#
# Canary: a real majority sample (class0_red, sg=0) — not a synthetic blank.
# Claim: majority samples are less privacy-vulnerable (smaller MIA signal, harder to audit),
# showing the defense's removal of minority-group samples is disproportionate.
#
# Step 1 (no defense): shows the majority canary is harder to audit → less privacy vulnerable.
# Step 2 (defense):    shows the defense does not flag/remove the majority canary.

echo "Master node: ${MASTER_ADDR}"

echo "=========================================="
echo "Step 1: Audit WITHOUT defense"
echo "=========================================="
srun --nodes=${N_NODES} --ntasks=${N_NODES} --ntasks-per-node=1 \
    torchrun \
    --nnodes=${N_NODES} \
    --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_id=${SLURM_JOB_ID}_1 \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    parallel_audit_fairness.py \
    --data_name colored_mnist \
    --model_name cnn \
    --n_reps ${N_REPS} \
    --n_epochs 100 \
    --lr 0.1 \
    --batch_size 4000 \
    --block_size 4000 \
    --aug_mult 1 \
    --epsilon "${EPSILON}" \
    --delta "${DELTA}" \
    --max_grad_norm "${MAX_GRAD_NORM}" \
    --sampling poisson \
    --target_type majority \
    --majority_pct 0.75 \
    --seed 0 \
    --fixed_init \
    --out "${OUTPUT_DIR}/no_defense"

echo "=========================================="
echo "Step 2: Audit WITH defense (unclipped grad norm, linf, no ascent)"
echo "=========================================="
srun --nodes=${N_NODES} --ntasks=${N_NODES} --ntasks-per-node=1 \
    torchrun \
    --nnodes=${N_NODES} \
    --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_id=${SLURM_JOB_ID}_2 \
    --rdzv_endpoint=${MASTER_ADDR}:$((MASTER_PORT + 1)) \
    parallel_audit_fairness.py \
    --data_name colored_mnist \
    --model_name cnn \
    --n_reps ${N_REPS} \
    --n_epochs 100 \
    --lr 0.1 \
    --batch_size 4000 \
    --block_size 4000 \
    --aug_mult 1 \
    --epsilon "${EPSILON}" \
    --delta "${DELTA}" \
    --max_grad_norm "${MAX_GRAD_NORM}" \
    --sampling poisson \
    --target_type majority \
    --majority_pct 0.75 \
    --seed 0 \
    --fixed_init \
    --defense \
    --defense_k 5 \
    --defense_score_fn grad_norm_unclipped \
    --defense_score_norm linf \
    --out "${OUTPUT_DIR}/defense"
