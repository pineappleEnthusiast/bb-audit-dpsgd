#!/bin/bash
set -e

MAJORITY_PCTS=(0.995 0.95 0.90 0.85 0.80 0.75)

for TASK_ID in $(seq 0 11); do
    SPLIT_IDX=$(( TASK_ID / 2 ))
    DEFENSE_IDX=$(( TASK_ID % 2 ))
    MAJORITY_PCT="${MAJORITY_PCTS[$SPLIT_IDX]}"

    if [ "$DEFENSE_IDX" -eq 0 ]; then
        DEFENSE_ARGS=""
        OUT_NAME="colored_mnist_maj${MAJORITY_PCT}_no_defense"
    else
        DEFENSE_ARGS="--defense --defense_k 5 --defense_score_fn grad_norm_unclipped --defense_score_norm linf"
        OUT_NAME="colored_mnist_maj${MAJORITY_PCT}_defense"
    fi

    echo "=========================================="
    echo "Task ${TASK_ID}: ${OUT_NAME}"
    echo "majority_pct: ${MAJORITY_PCT}  defense: ${DEFENSE_IDX}"
    echo "=========================================="

    torchrun --nnodes=1 --nproc_per_node=1 \
        --rdzv_backend=c10d \
        --rdzv_endpoint=localhost:29500 \
        fairness_audit.py \
        --data_name colored_mnist \
        --model_name cnn \
        --n_reps 4 \
        --n_epochs 100 \
        --lr 0.1 \
        --batch_size 4000 \
        --block_size 4000 \
        --aug_mult 1 \
        --epsilon -1 \
        --max_grad_norm -1 \
        --target_type blank \
        --seed 0 \
        --fixed_init \
        --sampling poisson \
        --holdout_audit \
        --majority_pct "$MAJORITY_PCT" \
        $DEFENSE_ARGS \
        --out "fairness_audits/$OUT_NAME"

    echo "Task ${TASK_ID} completed: ${OUT_NAME}"
done
