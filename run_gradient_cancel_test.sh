#!/bin/bash
set -e

OUTPUT_DIR="grad_cancel_test"
EPSILON=10
DELTA=1e-5
MAX_GRAD_NORM=1.0

# Private-regime gradient cancelling attack on MNIST/CNN.
#
# Design:
#   alpha=0.1 < max_grad_norm=1.0  -> Group A NOT clipped by DP; L∞=0.1
#                                      so defense does NOT filter Group A.
#   beta=9.0  > max_grad_norm=1.0  -> Group B clipped to 1.0 by DP; unclipped L∞=9.0
#                                      so defense easily detects and removes Group B.
#   Cancellation (DP-effective): n_A * alpha = n_B * min(beta, C)
#                                2000 * 0.1  = 200  * 1.0  = 200  ✓

echo "=========================================="
echo "Step 0: Check natural gradient norm distribution"
echo "=========================================="
python check_gradient_norms.py \
    --data_name mnist \
    --model_name cnn \
    --n_samples 500 \
    --device cuda:0

echo "=========================================="
echo "Step 1: Generate gradient cancelling canaries"
echo "=========================================="
python generate_gradient_cancelling_attack.py \
    --model_name cnn \
    --data_name mnist \
    --n_epochs 5 \
    --lr 3 \
    --max_grad_norm "${MAX_GRAD_NORM}" \
    --batch_size 4000 \
    --block_size 4000 \
    --n_group_a 2000 \
    --n_group_b 200 \
    --alpha 0.1 \
    --beta 9.0 \
    --defense_k 5 \
    --output_dir "${OUTPUT_DIR}" \
    --device cuda:0

echo "=========================================="
echo "Step 2: Audit WITHOUT defense (expect no gap)"
echo "=========================================="
torchrun --nnodes=1 --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:29500 \
    parallel_audit_multi_canary.py \
    --data_name mnist \
    --model_name cnn \
    --n_reps 8 \
    --n_epochs 30 \
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
    --gradient_space_score_fn hot_param \
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
    --data_name mnist \
    --model_name cnn \
    --n_reps 8 \
    --n_epochs 30 \
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
    --gradient_space_score_fn hot_param \
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
