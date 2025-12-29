#!/bin/bash

# Test version of Gradient Ascent Ablation Study
# Single GPU, reduced epochs/reps for quick testing in idev session

# Test settings
N_REPS=2
N_EPOCHS=1
TARGET_TYPE="blank"
DELTA=1e-5
DEFENSE_K=5
DEFENSE_SCORE_FN="grad_norm"
DEFENSE_SCORE_NORM="l2"
MAX_GRAD_NORM=1.0

echo "=========================================="
echo "Gradient Ascent Ablation Test (Single GPU)"
echo "=========================================="
echo "Settings: n_reps=2, n_epochs=1"
echo ""

#############################################
# MNIST Experiments
#############################################
echo "=========================================="
echo "MNIST + CNN Experiments"
echo "=========================================="

# Condition 1: NO DEFENSE
echo "Running MNIST - No defense - ε=2..."
torchrun --nproc_per_node=1 parallel_audit_model.py \
    --data_name mnist \
    --model_name cnn \
    --n_df 5000 \
    --n_reps $N_REPS \
    --n_epochs $N_EPOCHS \
    --lr 3 \
    --max_grad_norm $MAX_GRAD_NORM \
    --epsilon 2.0 \
    --delta $DELTA \
    --target_type $TARGET_TYPE \
    --batch_size 4000 \
    --block_size 4000 \
    --seed 0 \
    --aug_mult 1 \
    --out test_mnist_no_defense

echo "✓ MNIST - No defense completed"

# Condition 2: DEFENSE WITHOUT GRADIENT ASCENT
echo "Running MNIST - Defense (no ascent) - ε=2..."
torchrun --nproc_per_node=1 parallel_audit_model.py \
    --data_name mnist \
    --model_name cnn \
    --n_df 5000 \
    --n_reps $N_REPS \
    --n_epochs $N_EPOCHS \
    --lr 3 \
    --max_grad_norm $MAX_GRAD_NORM \
    --epsilon 2.0 \
    --delta $DELTA \
    --target_type $TARGET_TYPE \
    --batch_size 4000 \
    --block_size 4000 \
    --seed 0 \
    --aug_mult 1 \
    --defense \
    --defense_k $DEFENSE_K \
    --defense_score_fn $DEFENSE_SCORE_FN \
    --defense_score_norm $DEFENSE_SCORE_NORM \
    --out test_mnist_defense_no_ascent

echo "✓ MNIST - Defense (no ascent) completed"

# Condition 3: DEFENSE WITH GRADIENT ASCENT
echo "Running MNIST - Defense (with ascent) - ε=2..."
torchrun --nproc_per_node=1 parallel_audit_model.py \
    --data_name mnist \
    --model_name cnn \
    --n_df 5000 \
    --n_reps $N_REPS \
    --n_epochs $N_EPOCHS \
    --lr 3 \
    --max_grad_norm $MAX_GRAD_NORM \
    --epsilon 2.0 \
    --delta $DELTA \
    --target_type $TARGET_TYPE \
    --batch_size 4000 \
    --block_size 4000 \
    --seed 0 \
    --aug_mult 1 \
    --defense \
    --defense_k $DEFENSE_K \
    --defense_apply_ascent \
    --defense_score_fn $DEFENSE_SCORE_FN \
    --defense_score_norm $DEFENSE_SCORE_NORM \
    --out test_mnist_defense_with_ascent

echo "✓ MNIST - Defense (with ascent) completed"

#############################################
# CIFAR-10 Experiments
#############################################
echo "=========================================="
echo "CIFAR-10 + CNN Experiments"
echo "=========================================="

# Condition 1: NO DEFENSE
echo "Running CIFAR-10 - No defense - ε=10..."
torchrun --nproc_per_node=1 parallel_audit_model.py \
    --data_name cifar10 \
    --model_name cnn \
    --n_df 5000 \
    --n_reps $N_REPS \
    --n_epochs $N_EPOCHS \
    --lr 0.5 \
    --max_grad_norm $MAX_GRAD_NORM \
    --epsilon 10.0 \
    --delta $DELTA \
    --target_type $TARGET_TYPE \
    --batch_size 3125 \
    --block_size 3125 \
    --seed 0 \
    --aug_mult 1 \
    --out test_cifar10_no_defense

echo "✓ CIFAR-10 - No defense completed"

# Condition 2: DEFENSE WITHOUT GRADIENT ASCENT
echo "Running CIFAR-10 - Defense (no ascent) - ε=10..."
torchrun --nproc_per_node=1 parallel_audit_model.py \
    --data_name cifar10 \
    --model_name cnn \
    --n_df 5000 \
    --n_reps $N_REPS \
    --n_epochs $N_EPOCHS \
    --lr 0.5 \
    --max_grad_norm $MAX_GRAD_NORM \
    --epsilon 10.0 \
    --delta $DELTA \
    --target_type $TARGET_TYPE \
    --batch_size 3125 \
    --block_size 3125 \
    --seed 0 \
    --aug_mult 1 \
    --defense \
    --defense_k $DEFENSE_K \
    --defense_score_fn $DEFENSE_SCORE_FN \
    --defense_score_norm $DEFENSE_SCORE_NORM \
    --out test_cifar10_defense_no_ascent

echo "✓ CIFAR-10 - Defense (no ascent) completed"

# Condition 3: DEFENSE WITH GRADIENT ASCENT
echo "Running CIFAR-10 - Defense (with ascent) - ε=10..."
torchrun --nproc_per_node=1 parallel_audit_model.py \
    --data_name cifar10 \
    --model_name cnn \
    --n_df 5000 \
    --n_reps $N_REPS \
    --n_epochs $N_EPOCHS \
    --lr 0.5 \
    --max_grad_norm $MAX_GRAD_NORM \
    --epsilon 10.0 \
    --delta $DELTA \
    --target_type $TARGET_TYPE \
    --batch_size 3125 \
    --block_size 3125 \
    --seed 0 \
    --aug_mult 1 \
    --defense \
    --defense_k $DEFENSE_K \
    --defense_apply_ascent \
    --defense_score_fn $DEFENSE_SCORE_FN \
    --defense_score_norm $DEFENSE_SCORE_NORM \
    --out test_cifar10_defense_with_ascent

echo "✓ CIFAR-10 - Defense (with ascent) completed"

#############################################
# Purchase Experiments
#############################################
echo "=========================================="
echo "Purchase + MLP Experiments"
echo "=========================================="

# Condition 1: NO DEFENSE
echo "Running Purchase - No defense - ε=6..."
torchrun --nproc_per_node=1 parallel_audit_model.py \
    --data_name purchase \
    --model_name mlp \
    --n_df 100 \
    --n_reps $N_REPS \
    --n_epochs $N_EPOCHS \
    --lr 0.5 \
    --max_grad_norm $MAX_GRAD_NORM \
    --epsilon 6.0 \
    --delta $DELTA \
    --target_type $TARGET_TYPE \
    --batch_size 10 \
    --block_size 10 \
    --seed 0 \
    --aug_mult 1 \
    --out test_purchase_no_defense

echo "✓ Purchase - No defense completed"

# Condition 2: DEFENSE WITHOUT GRADIENT ASCENT
echo "Running Purchase - Defense (no ascent) - ε=6..."
torchrun --nproc_per_node=1 parallel_audit_model.py \
    --data_name purchase \
    --model_name mlp \
    --n_df 100 \
    --n_reps $N_REPS \
    --n_epochs $N_EPOCHS \
    --lr 0.5 \
    --max_grad_norm $MAX_GRAD_NORM \
    --epsilon 6.0 \
    --delta $DELTA \
    --target_type $TARGET_TYPE \
    --batch_size 10 \
    --block_size 10 \
    --seed 0 \
    --aug_mult 1 \
    --defense \
    --defense_k $DEFENSE_K \
    --defense_score_fn $DEFENSE_SCORE_FN \
    --defense_score_norm $DEFENSE_SCORE_NORM \
    --out test_purchase_defense_no_ascent

echo "✓ Purchase - Defense (no ascent) completed"

# Condition 3: DEFENSE WITH GRADIENT ASCENT
echo "Running Purchase - Defense (with ascent) - ε=6..."
torchrun --nproc_per_node=1 parallel_audit_model.py \
    --data_name purchase \
    --model_name mlp \
    --n_df 100 \
    --n_reps $N_REPS \
    --n_epochs $N_EPOCHS \
    --lr 0.5 \
    --max_grad_norm $MAX_GRAD_NORM \
    --epsilon 6.0 \
    --delta $DELTA \
    --target_type $TARGET_TYPE \
    --batch_size 10 \
    --block_size 10 \
    --seed 0 \
    --aug_mult 1 \
    --defense \
    --defense_k $DEFENSE_K \
    --defense_apply_ascent \
    --defense_score_fn $DEFENSE_SCORE_FN \
    --defense_score_norm $DEFENSE_SCORE_NORM \
    --out test_purchase_defense_with_ascent

echo "✓ Purchase - Defense (with ascent) completed"

echo "=========================================="
echo "All Test Experiments Completed!"
echo "=========================================="
echo "Summary:"
echo "  - MNIST:    3 experiments at ε=2  (2 reps, 1 epoch each)"
echo "  - CIFAR-10: 3 experiments at ε=10 (2 reps, 1 epoch each)"
echo "  - Purchase: 3 experiments at ε=6  (2 reps, 1 epoch each)"
echo "  - Total: 9 test experiments"
echo ""
echo "Results saved in test_* directories"
echo "=========================================="
