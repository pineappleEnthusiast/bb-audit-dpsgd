#!/bin/bash

# Test script to compare all 25 scoring functions on MNIST with blank canary
# Settings: n_df=10, n_reps=2, fit_world_only=in

# Common parameters
DATA_NAME="mnist"
MODEL_NAME="cnn"
N_DF=10
N_REPS=2
N_EPOCHS=5
LR=0.001
MAX_GRAD_NORM=1.0
EPSILON=10.0
DELTA=1e-5
TARGET_TYPE="blank"
BATCH_SIZE=256
BLOCK_SIZE=256
FIT_WORLD_ONLY="in"
OUT_DIR="test_results"

# List of all 25 scoring functions
SCORING_FUNCTIONS=(
    "grad_norm"
    "grad_norm_x_loss"
    "grad_norm_percentile"
    "grad_dir_volatility"
    "rand_proj_var"
    "maxmin_proj_ratio"
    "gradient_rank"
    "grad_accel"
    "grad_jerk"
    "norm_x_dir_uniqueness"
    "alignment_with_rand_proj"
    "gradient_sparsity"
    "gradient_kurtosis"
    "grad_dir_change_rate"
    "norm_x_trajectory_orth"
    "gradient_scatter"
    "fisher"
    "loss"
    "loss_momentum"
    "loss_volatility"
    "inv_confidence"
    "prediction_margin"
    "pred_entropy"
    "cos_update"
    "cos_theta0"
)

echo "Testing ${#SCORING_FUNCTIONS[@]} scoring functions on MNIST"
echo "Settings: n_df=$N_DF, n_reps=$N_REPS, n_epochs=$N_EPOCHS, fit_world_only=$FIT_WORLD_ONLY"
echo ""

# Run each scoring function
for score_fn in "${SCORING_FUNCTIONS[@]}"; do
    echo "=========================================="
    echo "Testing: $score_fn"
    echo "=========================================="
    
    torchrun --nproc_per_node=1 parallel_audit_model.py \
        --data_name $DATA_NAME \
        --model_name $MODEL_NAME \
        --n_df $N_DF \
        --n_reps $N_REPS \
        --n_epochs $N_EPOCHS \
        --lr $LR \
        --max_grad_norm $MAX_GRAD_NORM \
        --epsilon $EPSILON \
        --delta $DELTA \
        --target_type $TARGET_TYPE \
        --batch_size $BATCH_SIZE \
        --block_size $BLOCK_SIZE \
        --fit_world_only $FIT_WORLD_ONLY \
        --defense \
        --defense_score_fn $score_fn \
        --out "${OUT_DIR}/${score_fn}"
    
    if [ $? -eq 0 ]; then
        echo "✓ $score_fn completed successfully"
    else
        echo "✗ $score_fn failed"
    fi
    echo ""
done

echo "=========================================="
echo "All tests completed!"
echo "Results saved in: $OUT_DIR/"
echo "=========================================="
