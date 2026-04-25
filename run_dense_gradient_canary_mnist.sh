#!/bin/bash
set -e

OUTPUT_DIR="dense_grad_canary_mnist_private"
EPSILON=10
DELTA=1e-5
MAX_GRAD_NORM=1.0

# Dense unit-vector gradient canary attack on MNIST/CNN (private regime).
#
# Design:
#   v = random unit vector in ℝ^D (D=CNN param count), scaled to L2=max_grad_norm=1.0
#   L∞(v) ≈ 1/√D << defense threshold → defense does NOT filter this canary
#   Score = dot(Δθ, v): large when canary is in training set, ~0 otherwise
#   Uses full DP budget (L2=1.0), evading the L∞ defense entirely.

echo "=========================================="
echo "Step 0: Check natural gradient norm distribution"
echo "=========================================="
python check_gradient_norms.py \
    --data_name mnist \
    --model_name cnn \
    --n_samples 500 \
    --device cuda:0

echo "=========================================="
echo "Step 1: Generate dense unit-vector gradient canary"
echo "=========================================="
python generate_dense_gradient_canary.py \
    --model_name cnn \
    --data_name mnist \
    --scale "${MAX_GRAD_NORM}" \
    --seed 0 \
    --output_dir "${OUTPUT_DIR}" \
    --device cuda:0

echo "=========================================="
echo "Step 2: Audit WITHOUT defense (expect gap)"
echo "=========================================="
torchrun --nnodes=1 --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:29500 \
    parallel_audit_multi_canary.py \
    --data_name mnist \
    --model_name cnn \
    --n_reps 4 \
    --n_epochs 100 \
    --lr 3 \
    --batch_size 4000 \
    --block_size 4000 \
    --epsilon "${EPSILON}" \
    --delta "${DELTA}" \
    --max_grad_norm "${MAX_GRAD_NORM}" \
    --sampling poisson \
    --holdout_audit \
    --gradient_space_canary_pt "${OUTPUT_DIR}/gradient_space_canaries.pt" \
    --target_type gradient_space_canary \
    --gradient_space_score_fn dot_product \
    --seed 0 \
    --fixed_init \
    --out "${OUTPUT_DIR}/audit_no_defense"

echo "=========================================="
echo "Step 3: Audit WITH defense (gap should persist — canary evades L∞ filter)"
echo "=========================================="
torchrun --nnodes=1 --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:29500 \
    parallel_audit_multi_canary.py \
    --data_name mnist \
    --model_name cnn \
    --n_reps 4 \
    --n_epochs 100 \
    --lr 3 \
    --batch_size 4000 \
    --block_size 4000 \
    --epsilon "${EPSILON}" \
    --delta "${DELTA}" \
    --max_grad_norm "${MAX_GRAD_NORM}" \
    --sampling poisson \
    --holdout_audit \
    --gradient_space_canary_pt "${OUTPUT_DIR}/gradient_space_canaries.pt" \
    --target_type gradient_space_canary \
    --gradient_space_score_fn dot_product \
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
