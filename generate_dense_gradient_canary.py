"""
Generate a dense unit-vector gradient space canary.

Creates a single canary whose per-sample gradient is a random unit vector in parameter space,
scaled to match max_grad_norm (L2 = scale, L∞ ≈ 1/√D ≈ 0.002 for D=245k on Purchase MLP).

This canary is invisible to the L∞-based defense while using the full DP privacy budget,
making it a stronger adaptive attack than gradient cancelling Group A (which wastes 90% of the budget).

Scoring: dot(Δθ, v) where v is the saved direction and Δθ = final_params - init_params.
Use --gradient_space_score_fn dot_product in parallel_audit_multi_canary.py.
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

from models import Models
from utils.data import load_data


def xavier_init_model(model):
    def init_weights(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)
    model.apply(init_weights)


def main():
    parser = argparse.ArgumentParser(description='Generate dense unit-vector gradient space canary')
    parser.add_argument('--model_name', type=str, required=True, choices=['mlp', 'cnn', 'wideresnet', 'lstm'])
    parser.add_argument('--data_name', type=str, required=True)
    parser.add_argument('--out_dim', type=int, default=None, help='Output dimension (e.g. 100 for Purchase)')
    parser.add_argument('--scale', type=float, default=1.0,
                        help='L2 norm of the canary gradient. Set to max_grad_norm for full DP budget.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    # Load a small slice of data just to get input shape and out_dim
    X, y, out_dim_data = load_data(args.data_name, n_df=None)
    out_dim = args.out_dim if args.out_dim is not None else out_dim_data

    # Build model to get parameter structure
    model = Models[args.model_name](X.shape, out_dim=out_dim).to(device)
    xavier_init_model(model)
    model.eval()

    # Count total parameters and record layout
    param_names = [name for name, _ in model.named_parameters()]
    param_shapes = {name: p.shape for name, p in model.named_parameters()}
    param_sizes = {name: p.numel() for name, p in model.named_parameters()}
    total_params = sum(param_sizes.values())

    print(f"Model: {args.model_name}, total params D={total_params}")
    print(f"Expected L∞ of unit vector ≈ 1/√D = {1.0 / (total_params ** 0.5):.6f}")

    # Sample a random unit vector v ∈ ℝ^D, scaled to args.scale
    v_flat = torch.randn(total_params, generator=torch.Generator().manual_seed(args.seed))
    v_flat = v_flat / v_flat.norm(p=2) * args.scale

    l2_norm = float(v_flat.norm(p=2).item())
    linf_norm = float(v_flat.abs().max().item())
    print(f"Canary gradient: L2={l2_norm:.6f}, L∞={linf_norm:.6f} (scale={args.scale})")
    print(f"Defense threshold context: L∞={linf_norm:.6f} is ~{linf_norm / (1.0 / total_params**0.5):.1f}× the expected 1/√D")

    # Reshape v_flat back into a gradient dict, adding batch dim (1, ...)
    grad_dict = {}
    offset = 0
    for name in param_names:
        size = param_sizes[name]
        shape = param_shapes[name]
        chunk = v_flat[offset: offset + size].reshape(shape)
        grad_dict[name] = chunk.unsqueeze(0).cpu()  # shape (1, *shape)
        offset += size

    assert offset == total_params

    # Save in format expected by parallel_audit_multi_canary.py
    output_path = output_dir / 'gradient_space_canaries.pt'
    torch.save({
        'gradients': [grad_dict],   # list of one gradient dict
        'direction': v_flat.cpu(),  # 1D tensor for dot_product scoring
        'n_canaries': 1,
        'scale': args.scale,
        'l2_norm': l2_norm,
        'linf_norm': linf_norm,
        'total_params': total_params,
        'model_name': args.model_name,
        'seed': args.seed,
    }, output_path)
    print(f"\nSaved dense gradient canary to {output_path}")
    print(f"  gradients: 1 entry (grad_dict with {len(grad_dict)} parameter tensors, each shape (1, ...))")
    print(f"  direction: 1D tensor of shape ({total_params},) for dot_product scoring")


if __name__ == '__main__':
    main()
