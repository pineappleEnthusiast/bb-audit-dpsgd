"""
Generate gradient cancelling attack with multi-canary gradient space approach.

This script:
1. Creates two groups of canaries (A and B) with opposite gradient directions
2. Group A: n_group_a canaries with norm alpha (positive gradients)
3. Group B: n_group_b canaries with norm beta (negative gradients)
4. Constraint: n_group_a * alpha = n_group_b * beta (gradients cancel out)
5. Saves canaries to .pt files for use in auditing
"""

import os
import torch
import torch.nn as nn
import numpy as np
import argparse
from pathlib import Path

from models import Models


def create_1hot_gradient(model, hot_index, norm_value, device='cuda'):
    """
    Create a 1-hot gradient vector with specified norm at given index.
    
    Args:
        model: PyTorch model to get parameter structure
        hot_index: Global index where the non-zero value is placed
        norm_value: L∞ norm value (magnitude of the single non-zero element)
        device: Device to place tensors on
    
    Returns:
        Dictionary of gradient tensors with batch dimension
    """
    params = {}
    total_elements = 0

    for name, param in model.named_parameters():
        if param.requires_grad:
            num_elements = param.numel()
            params[name] = {
                'param': param,
                'start_idx': total_elements,
                'end_idx': total_elements + num_elements,
                'shape': param.shape
            }
            total_elements += num_elements

    if total_elements == 0:
        raise ValueError("Model has no trainable parameters")

    crafted_grad = {}
    for name, info in params.items():
        param = info['param']
        if param.requires_grad:
            grad = torch.zeros_like(param)
            if info['start_idx'] <= hot_index < info['end_idx']:
                local_idx = hot_index - info['start_idx']
                flat_grad = grad.view(-1)
                flat_grad[local_idx] = norm_value
                grad = flat_grad.view(info['shape'])
            crafted_grad[name] = grad.unsqueeze(0)
        else:
            crafted_grad[name] = torch.zeros_like(param).unsqueeze(0)

    return crafted_grad


def main():
    parser = argparse.ArgumentParser(description='Generate gradient cancelling attack canaries')
    parser.add_argument('--model_name', type=str, default='cnn', choices=list(Models.keys()),
                        help='model architecture')
    parser.add_argument('--data_name', type=str, default='cifar10',
                        help='dataset name (used for model input shape)')
    parser.add_argument('--out_dim', type=int, default=10,
                        help='output dimension (number of classes)')
    parser.add_argument('--n_group_a', type=int, default=500,
                        help='number of canaries in group A (positive gradients)')
    parser.add_argument('--n_group_b', type=int, default=600,
                        help='number of canaries in group B (negative gradients)')
    parser.add_argument('--alpha', type=float, default=1000.0,
                        help='L∞ norm magnitude for group A canaries')
    parser.add_argument('--output_dir', type=str, default='gradient_cancelling_canaries',
                        help='output directory for canary files')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='device to use')
    
    args = parser.parse_args()

    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute beta such that n_group_a * alpha = n_group_b * beta
    beta = (args.n_group_a * args.alpha) / args.n_group_b
    print(f"Group A: {args.n_group_a} canaries with norm alpha={args.alpha:.6f}")
    print(f"Group B: {args.n_group_b} canaries with norm beta={beta:.6f}")
    print(f"Constraint check: {args.n_group_a} * {args.alpha:.6f} = {args.n_group_b} * {beta:.6f}")
    print(f"  LHS: {args.n_group_a * args.alpha:.6f}")
    print(f"  RHS: {args.n_group_b * beta:.6f}")

    # Create a dummy model to get parameter structure
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    
    if args.model_name == 'lstm':
        dummy_model = Models[args.model_name](vocab_size=args.out_dim, out_dim=args.out_dim)
    else:
        dummy_model = Models[args.model_name]((1, 3, 32, 32), out_dim=args.out_dim)

    # Count total parameters
    total_params = sum(p.numel() for p in dummy_model.parameters() if p.requires_grad)
    print(f"Total model parameters: {total_params}")

    # Generate group A canaries (positive gradients)
    print(f"\nGenerating {args.n_group_a} group A canaries...")
    group_a_canaries = []
    for i in range(args.n_group_a):
        # Random 1-hot index for each canary
        hot_index = np.random.randint(0, total_params)
        grad = create_1hot_gradient(dummy_model, hot_index, args.alpha, device)
        group_a_canaries.append(grad)
        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{args.n_group_a} canaries")

    # Generate group B canaries (negative gradients)
    print(f"\nGenerating {args.n_group_b} group B canaries...")
    group_b_canaries = []
    for i in range(args.n_group_b):
        # Random 1-hot index for each canary
        hot_index = np.random.randint(0, total_params)
        grad = create_1hot_gradient(dummy_model, hot_index, -beta, device)
        group_b_canaries.append(grad)
        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{args.n_group_b} canaries")

    # Convert gradient dictionaries to tensor format for parallel_audit_multi_canary.py
    # Each canary is a dict of parameter gradients; we need to flatten them into a single tensor
    def flatten_gradient_dict(grad_dict):
        """Flatten a gradient dictionary into a single 1D tensor"""
        flat_grads = []
        for name in sorted(grad_dict.keys()):
            g = grad_dict[name].squeeze(0).view(-1)
            flat_grads.append(g)
        return torch.cat(flat_grads)
    
    # Flatten all canaries
    group_a_tensors = torch.stack([flatten_gradient_dict(g) for g in group_a_canaries])
    group_b_tensors = torch.stack([flatten_gradient_dict(g) for g in group_b_canaries])
    
    # Create labels (all 0 for group A, all 0 for group B - they're just canaries)
    group_a_labels = torch.zeros(args.n_group_a, dtype=torch.long)
    group_b_labels = torch.zeros(args.n_group_b, dtype=torch.long)
    
    # Save group A canaries in format expected by parallel_audit_multi_canary.py
    group_a_file = output_dir / 'group_a_canaries.pt'
    torch.save({
        'canaries': group_a_tensors,
        'audit_labels': group_a_labels,
        'norm': args.alpha,
        'group': 'A'
    }, group_a_file)
    print(f"\nSaved {args.n_group_a} group A canaries to {group_a_file}")

    # Save group B canaries in format expected by parallel_audit_multi_canary.py
    group_b_file = output_dir / 'group_b_canaries.pt'
    torch.save({
        'canaries': group_b_tensors,
        'audit_labels': group_b_labels,
        'norm': beta,
        'group': 'B'
    }, group_b_file)
    print(f"Saved {args.n_group_b} group B canaries to {group_b_file}")

    # Save metadata
    metadata_file = output_dir / 'metadata.pt'
    torch.save({
        'n_group_a': args.n_group_a,
        'n_group_b': args.n_group_b,
        'alpha': args.alpha,
        'beta': beta,
        'total_params': total_params,
        'model_name': args.model_name,
        'seed': args.seed
    }, metadata_file)
    print(f"Saved metadata to {metadata_file}")

    # Verify cancellation property
    print(f"\nVerification:")
    print(f"  Group A total gradient magnitude: {args.n_group_a} * {args.alpha:.6f} = {args.n_group_a * args.alpha:.6f}")
    print(f"  Group B total gradient magnitude: {args.n_group_b} * {beta:.6f} = {args.n_group_b * beta:.6f}")
    print(f"  Cancellation check: {abs(args.n_group_a * args.alpha + args.n_group_b * beta):.10f} (should be ~0)")


if __name__ == '__main__':
    main()
