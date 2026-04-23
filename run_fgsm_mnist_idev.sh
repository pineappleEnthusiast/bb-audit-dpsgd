#!/bin/bash
set -e

OUTPUT_DIR="fgsm_mnist_eps10"
EPSILON=10
DELTA=1e-5
MAX_GRAD_NORM=1.0
N_REPS=400
N_NODES=5
MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
MASTER_PORT=29500

# FGSM canary audit on MNIST/CNN, private regime (eps=10).
# Canary: adversarial example (FGSM, target = original_label + 1 mod 10).
# Expected: no-defense run shows gap; defense run shows gap shrinks / disappears.

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
    parallel_audit_model.py \
    --data_name mnist \
    --model_name cnn \
    --n_reps ${N_REPS} \
    --n_epochs 100 \
    --lr 3 \
    --batch_size 4000 \
    --block_size 4000 \
    --epsilon "${EPSILON}" \
    --delta "${DELTA}" \
    --max_grad_norm "${MAX_GRAD_NORM}" \
    --sampling poisson \
    --target_type fgsm \
    --seed 0 \
    --fixed_init \
    --holdout_audit \
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
    parallel_audit_model.py \
    --data_name mnist \
    --model_name cnn \
    --n_reps ${N_REPS} \
    --n_epochs 100 \
    --lr 3 \
    --batch_size 4000 \
    --block_size 4000 \
    --epsilon "${EPSILON}" \
    --delta "${DELTA}" \
    --max_grad_norm "${MAX_GRAD_NORM}" \
    --sampling poisson \
    --target_type fgsm \
    --seed 0 \
    --fixed_init \
    --holdout_audit \
    --defense \
    --defense_k 5 \
    --defense_score_fn grad_norm_unclipped \
    --defense_score_norm linf \
    --out "${OUTPUT_DIR}/defense"
