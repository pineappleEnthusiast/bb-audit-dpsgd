#!/bin/bash
set -e

OUTPUT_DIR="input_cancel_test_purchase"
EPSILON=10
DELTA=1e-5
MAX_GRAD_NORM=1.0

# Input-space cancelling canary attack on Purchase/MLP, private regime (eps=10).
#
# Key insight: for an MLP, the per-sample first-layer gradient is δ₁ ⊗ x, so
# input vectors directly determine the gradient direction.  No synthetic gradient
# dicts are needed — we design cancellation in input space.
#
# Design (1-hot in input feature space at hot_dim):
#   alpha=0.9,  n_group_a=2000  →  L∞ grad ∝ 0.9  (evades defence)
#   beta=9.0,   n_group_b=200   →  L∞ grad ∝ 9.0  (detected, removed)
#   Cancellation: 2000 * 0.9 = 200 * 9.0 = 1800 ✓
#
# Step 2 (no defence):  A + B in training → net gradient ≈ 0 → no MIA gap.
# Step 3 (defence):     B removed (high L∞) → A alone memorised → gap appears.

echo "=========================================="
echo "Step 1: Generate input cancelling canaries"
echo "=========================================="
python generate_input_cancelling_canaries.py \
    --model_name mlp \
    --data_name purchase \
    --out_dim 100 \
    --n_epochs 5 \
    --lr 10 \
    --batch_size 12143 \
    --n_group_a 2000 \
    --n_group_b 200 \
    --alpha 0.9 \
    --label 0 \
    --output_dir "${OUTPUT_DIR}" \
    --device cuda:0

echo "=========================================="
echo "Step 2: Audit WITHOUT defense (expect no gap)"
echo "=========================================="
torchrun --nnodes=1 --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:29500 \
    parallel_audit_multi_canary.py \
    --data_name purchase \
    --model_name mlp \
    --n_df 153801 \
    --n_reps 4 \
    --n_epochs 100 \
    --lr 10 \
    --batch_size 12143 \
    --block_size 12143 \
    --aug_mult 1 \
    --epsilon "${EPSILON}" \
    --delta "${DELTA}" \
    --max_grad_norm "${MAX_GRAD_NORM}" \
    --sampling poisson \
    --holdout_audit \
    --canary_pt "${OUTPUT_DIR}/input_cancelling_canaries.pt" \
    --target_type blank \
    --seed 0 \
    --fixed_init \
    --out "${OUTPUT_DIR}/audit_no_defense"

echo "=========================================="
echo "Step 3: Audit WITH defense (expect gap)"
echo "=========================================="
torchrun --nnodes=1 --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:29500 \
    parallel_audit_multi_canary.py \
    --data_name purchase \
    --model_name mlp \
    --n_df 153801 \
    --n_reps 4 \
    --n_epochs 100 \
    --lr 10 \
    --batch_size 12143 \
    --block_size 12143 \
    --aug_mult 1 \
    --epsilon "${EPSILON}" \
    --delta "${DELTA}" \
    --max_grad_norm "${MAX_GRAD_NORM}" \
    --sampling poisson \
    --holdout_audit \
    --canary_pt "${OUTPUT_DIR}/input_cancelling_canaries.pt" \
    --target_type blank \
    --seed 0 \
    --fixed_init \
    --defense \
    --defense_k 5 \
    --defense_score_fn grad_norm_unclipped \
    --defense_score_norm linf \
    --out "${OUTPUT_DIR}/audit_defense"

echo "=========================================="
echo "Step 4: Compare mean MIA scores"
echo "=========================================="
python compare_cancel_scores.py "${OUTPUT_DIR}"
