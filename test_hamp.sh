#!/bin/bash
# Quick test of hamp_defense.py with minimal settings

echo "Testing HAMP Defense with gamma=0.0 (baseline, no defense)"
python hamp_defense.py \
    --n_reps 10 \
    --n_epochs 2 \
    --gamma 0.0 \
    --batch_size 64 \
    --out exp_data/hamp_test

echo ""
echo "Testing HAMP Defense with gamma=0.8 (soft label defense)"
python hamp_defense.py \
    --n_reps 10 \
    --n_epochs 2 \
    --gamma 0.8 \
    --batch_size 64 \
    --out exp_data/hamp_test

echo ""
echo "Comparing results..."
python -c "
import numpy as np
import os

gamma0_path = 'exp_data/hamp_test/mnist_cnn_gamma0.0/emp_eps_loss.npy'
gamma08_path = 'exp_data/hamp_test/mnist_cnn_gamma0.8/emp_eps_loss.npy'

if os.path.exists(gamma0_path) and os.path.exists(gamma08_path):
    eps0 = np.load(gamma0_path)[0]
    eps08 = np.load(gamma08_path)[0]
    print(f'Empirical epsilon (gamma=0.0): {eps0:.4f}')
    print(f'Empirical epsilon (gamma=0.8): {eps08:.4f}')
    print(f'Privacy gain: {eps0 - eps08:.4f}')
    if eps08 < eps0:
        print('✓ Soft labels provide privacy protection!')
    else:
        print('✗ Soft labels did not reduce empirical epsilon')
else:
    print('Error: Output files not found')
"
