#!/bin/bash
set -e

OUTPUT_DIR="grad_cancel_test"

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
    --n_group_a 50 \
    --n_group_b 5 \
    --alpha 0.1 \
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
import numpy as np
for tag in ['no_defense', 'defense']:
    path = f'${OUTPUT_DIR}/audit_{tag}/results.npy'
    d = np.load(path, allow_pickle=True).item()
    scores = np.array(d['mia_scores'])
    labels = np.array(d['mia_labels'])
    mean_in  = scores[labels == 1].mean()
    mean_out = scores[labels == 0].mean()
    gap = mean_in - mean_out
    print(f'{tag:12s}: in={mean_in:.4f}  out={mean_out:.4f}  gap={gap:.4f}')
"
