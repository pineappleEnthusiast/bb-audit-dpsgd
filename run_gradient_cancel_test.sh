#!/bin/bash
set -e

OUTPUT_DIR="grad_cancel_test"

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
    --batch_size 4000 \
    --block_size 4000 \
    --n_group_a 2000 \
    --n_group_b 200 \
    --alpha 5.0 \
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
    --epsilon -1 \
    --max_grad_norm -1 \
    --sampling poisson \
    --holdout_audit \
    --gradient_space_canary_pt "${OUTPUT_DIR}/gradient_space_canaries.pt" \
    --target_type gradient_space_canary \
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
    --epsilon -1 \
    --max_grad_norm -1 \
    --sampling poisson \
    --holdout_audit \
    --gradient_space_canary_pt "${OUTPUT_DIR}/gradient_space_canaries.pt" \
    --target_type gradient_space_canary \
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
python -c "
import numpy as np, os
for tag in ['no_defense', 'defense']:
    d = '${OUTPUT_DIR}/audit_' + tag
    scores_in  = np.load(os.path.join(d, 'scores_in.npy'))
    scores_out = np.load(os.path.join(d, 'scores_out.npy'))
    mean_in  = scores_in.mean()
    mean_out = scores_out.mean()
    gap = mean_in - mean_out
    print(f'{tag:12s}: in={mean_in:.4f}  out={mean_out:.4f}  gap={gap:.4f}')
"
