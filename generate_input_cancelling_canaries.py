"""
Generate input-space cancelling canaries for Purchase/MLP.

For an MLP the per-sample gradient at the first layer is  δ₁ ⊗ x,  so the input
vector x directly determines the gradient direction in that layer.  Cancellation
can therefore be designed purely in input space:

  Group A : n_group_a canaries with input[hot_dim] = +alpha   (low  L∞ gradient → evades defence)
  Group B : n_group_b canaries with input[hot_dim] = -beta    (high L∞ gradient → removed by defence)
  Constraint: n_group_a * alpha = n_group_b * beta            (gradients cancel when both are in)

Without defence:  A + B in training → net first-layer gradient ≈ 0 → no MIA gap.
With defence:     Group B (large |input[hot_dim]|) is removed → A alone memorised → gap appears.

hot_dim is found by training briefly, then picking the input feature j with the
largest average |∂L/∂x_j| across a sample of training data.
"""

import argparse
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader

from models import Models
from utils.data import load_data


def xavier_init_model(model):
    import torch.nn as nn
    def _init(m):
        if isinstance(m, (nn.Linear, torch.nn.Conv2d)):
            torch.nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)
    model.apply(_init)


def train_briefly(model, X, y, device, n_epochs, lr, batch_size):
    """Quick SGD (no DP) training to get a meaningful model for hot-dim detection."""
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)
    for _ in range(n_epochs):
        for X_b, y_b in loader:
            opt.zero_grad()
            F.cross_entropy(model(X_b.to(device)), y_b.to(device)).backward()
            opt.step()


def find_hot_input_dim(model, X, y, device, n_samples=500):
    """Return the input feature index with the highest average |∂L/∂x_j|."""
    model.eval()
    X_s = X[:n_samples].clone().to(device).requires_grad_(True)
    y_s = y[:n_samples].to(device)
    F.cross_entropy(model(X_s), y_s).backward()
    avg_abs = X_s.grad.abs().mean(0)   # (input_dim,)
    hot = int(avg_abs.argmax().item())
    print(f"Hot input dim: {hot}  (avg |grad|={avg_abs[hot].item():.6f})")
    return hot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='mlp', choices=list(Models.keys()))
    parser.add_argument('--data_name', default='purchase')
    parser.add_argument('--out_dim', type=int, default=None)
    parser.add_argument('--n_epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=10.0)
    parser.add_argument('--batch_size', type=int, default=12143)
    parser.add_argument('--n_group_a', type=int, default=2000,
                        help='canaries in group A (+alpha, evades defence)')
    parser.add_argument('--n_group_b', type=int, default=200,
                        help='canaries in group B (-beta, detected by defence)')
    parser.add_argument('--alpha', type=float, default=0.9,
                        help='input magnitude for group A (low, evades L∞ defence)')
    parser.add_argument('--beta', type=float, default=None,
                        help='input magnitude for group B (default: n_group_a*alpha/n_group_b)')
    parser.add_argument('--label', type=int, default=0,
                        help='audit label assigned to all canaries')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y, out_dim_data = load_data(args.data_name, n_df=None)
    out_dim = args.out_dim if args.out_dim is not None else out_dim_data
    input_dim = X.shape[1]

    print(f"Dataset: {args.data_name}, N={len(X)}, input_dim={input_dim}, out_dim={out_dim}")

    model = Models[args.model_name](X.shape, out_dim=out_dim).to(device)
    xavier_init_model(model)

    print(f"Training briefly ({args.n_epochs} epochs) to identify hot input dimension …")
    train_briefly(model, X, y, device, args.n_epochs, args.lr, args.batch_size)

    hot_dim = find_hot_input_dim(model, X, y, device)

    beta = args.beta if args.beta is not None else (args.n_group_a * args.alpha / args.n_group_b)
    print(f"\nGroup A: {args.n_group_a} canaries, input[{hot_dim}] = +{args.alpha:.4f}")
    print(f"Group B: {args.n_group_b} canaries, input[{hot_dim}] = -{beta:.4f}")
    print(f"Cancellation check: {args.n_group_a} * {args.alpha} = {args.n_group_b} * {beta:.4f}  "
          f"→ {args.n_group_a * args.alpha:.4f} vs {args.n_group_b * beta:.4f}")

    # Build group A: 1-hot in input space at hot_dim with value +alpha
    x_a = torch.zeros(input_dim)
    x_a[hot_dim] = args.alpha
    X_a = x_a.unsqueeze(0).expand(args.n_group_a, -1).clone()
    y_a = torch.full((args.n_group_a,), args.label, dtype=torch.long)

    # Build group B: 1-hot in input space at hot_dim with value -beta
    x_b = torch.zeros(input_dim)
    x_b[hot_dim] = -beta
    X_b = x_b.unsqueeze(0).expand(args.n_group_b, -1).clone()
    y_b = torch.full((args.n_group_b,), args.label, dtype=torch.long)

    X_canary = torch.vstack([X_a, X_b])
    y_canary = torch.cat([y_a, y_b])

    out_path = output_dir / 'input_cancelling_canaries.pt'
    torch.save({
        'canaries': X_canary,
        'audit_labels': y_canary,
        'hot_dim': hot_dim,
        'alpha': args.alpha,
        'beta': beta,
        'n_group_a': args.n_group_a,
        'n_group_b': args.n_group_b,
        'input_dim': input_dim,
    }, out_path)
    print(f"\nSaved {len(X_canary)} canaries to {out_path}")
    print(f"  shape: {tuple(X_canary.shape)}, labels: {y_canary.unique().tolist()}")


if __name__ == '__main__':
    main()
