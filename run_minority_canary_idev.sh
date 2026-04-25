#!/bin/bash
set -e

OUTPUT_DIR="minority_canary_colored_mnist_eps10"
EPSILON=10
DELTA=1e-5
MAX_GRAD_NORM=1.0
N_REPS=200
N_NODES=5
MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
MASTER_PORT=29500

# Minority-canary audit on Colored MNIST (75% majority split), private regime (eps=10).
#
# Canary: a real minority sample (class0_blue, sg=1) — not a synthetic blank.
# Claim: minority samples are more privacy-vulnerable (larger MIA signal, easier to audit),
# justifying the defense's disproportionate removal of minority-group samples.
#
# Step 1 (no defense): shows the minority canary is auditable → privacy vulnerable.
# Step 2 (defense):    shows the defense correctly identifies and removes the canary.

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
    --target_type minority \
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
    --target_type minority \
    --majority_pct 0.75 \
    --seed 0 \
    --fixed_init \
    --defense \
    --defense_k 5 \
    --defense_score_fn grad_norm_unclipped \
    --defense_score_norm linf \
    --out "${OUTPUT_DIR}/defense"
