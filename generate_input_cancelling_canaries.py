"""
Generate input-space cancelling canaries for Purchase/MLP.

Both groups use the same random dense unit vector but opposite signs, so their
total input contribution sums to zero:

  Group A: n_group_a canaries, x = +random_dense * alpha, label = label_a
           Larger per-sample norm (alpha > beta) → gets filtered by defense
  Group B: n_group_b canaries, x = -random_dense * beta,  label = label_b
           Smaller per-sample norm → survives defense, provides audit signal

Cancellation constraint: n_group_a * alpha = n_group_b * beta
  => beta = n_group_a * alpha / n_group_b  (computed automatically if not passed)
  => alpha > beta since n_group_b > n_group_a

Both alpha and beta are >> regular data gradient norms, so both groups have
absurdly large gradient norms relative to normal samples.

Without defence:  A + B cancel → net gradient ≈ 0 → no MIA signal.
With defence:     Group A (large norm) filtered → Group B alone memorised → gap appears.
"""

import argparse
import torch
import numpy as np
from pathlib import Path

from utils.data import load_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', default='purchase')
    parser.add_argument('--n_group_a', type=int, default=500,
                        help='canaries in group A (larger norm, gets filtered by defense)')
    parser.add_argument('--n_group_b', type=int, default=1000,
                        help='canaries in group B (smaller norm, survives defense, provides audit signal)')
    parser.add_argument('--alpha', type=float, default=20.0,
                        help='per-sample input magnitude for group A (>> regular data norms)')
    parser.add_argument('--beta', type=float, default=None,
                        help='per-sample input magnitude for group B (default: n_group_a*alpha/n_group_b)')
    parser.add_argument('--label', type=int, default=0,
                        help='label for group A canaries')
    parser.add_argument('--label_b', type=int, default=1,
                        help='label for group B canaries (different from --label for opposing loss signal)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y, _ = load_data(args.data_name, n_df=None)
    input_dim = X.shape[1]

    beta = args.beta if args.beta is not None else (args.n_group_a * args.alpha / args.n_group_b)

    print(f"Dataset: {args.data_name}, N={len(X)}, input_dim={input_dim}")
    print(f"\nInput-Space Cancelling Canary Design:")
    print(f"  Group A: {args.n_group_a} × (+random_dense × {args.alpha:.4f}), label={args.label}  ← filtered by defense")
    print(f"  Group B: {args.n_group_b} × (-random_dense × {beta:.4f}),  label={args.label_b}  ← survives defense")
    print(f"  Cancellation: {args.n_group_a}×{args.alpha:.4f} = {args.n_group_b}×{beta:.4f}  ({args.n_group_a*args.alpha:.1f} vs {args.n_group_b*beta:.1f})")
    print(f"  α/β = {args.alpha/beta:.1f}x")

    random_dense = torch.randn(input_dim)
    random_dense = random_dense / random_dense.norm(p=2)

    x_a = random_dense * args.alpha
    X_a = x_a.unsqueeze(0).expand(args.n_group_a, -1).clone()
    y_a = torch.full((args.n_group_a,), args.label, dtype=torch.long)

    x_b = -random_dense * beta
    X_b = x_b.unsqueeze(0).expand(args.n_group_b, -1).clone()
    y_b = torch.full((args.n_group_b,), args.label_b, dtype=torch.long)

    X_canary = torch.vstack([X_a, X_b])
    y_canary = torch.cat([y_a, y_b])

    out_path = output_dir / 'input_cancelling_canaries.pt'
    torch.save({
        'canaries': X_canary,
        'audit_labels': y_canary,
        'alpha': args.alpha,
        'beta': beta,
        'n_group_a': args.n_group_a,
        'n_group_b': args.n_group_b,
        'label_a': args.label,
        'label_b': args.label_b,
        'input_dim': input_dim,
    }, out_path)
    print(f"\nSaved {len(X_canary)} canaries to {out_path}")
    print(f"  shape: {tuple(X_canary.shape)}, labels: {y_canary.unique().tolist()}")


if __name__ == '__main__':
    main()
