"""
Generate gradient-matched cancelling canaries.

Optimizes x_A and x_B so that their parameter gradients at epoch 0 (fixed init θ₀)
satisfy the cancellation scheme:

  n_A * grad(x_A, y_A; θ₀) + n_B * grad(x_B, y_B; θ₀) ≈ 0

with:
  ‖grad(x_A, ...)‖_∞ = target_linf          (large → always in top-k, gets filtered)
  ‖grad(x_B, ...)‖_∞ = target_linf * n_A/n_B (smaller → survives defense, audit signal)

The init state θ₀ is embedded in the saved .pt so the audit script uses the same init.
"""

import argparse
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

from utils.data import load_data
from models import Models
from utils.training import xavier_init_model


def _grad_flat(model, x, y_label, device, create_graph=False):
    model.zero_grad()
    logits = model(x.unsqueeze(0))
    loss = F.cross_entropy(logits, torch.tensor([y_label], device=device))
    params = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(loss, params, create_graph=create_graph)
    return torch.cat([g.flatten() for g in grads])


def optimize_input(model, g_target, y_label, input_dim, device, n_steps, lr):
    x = torch.zeros(input_dim, device=device, requires_grad=True)
    opt = torch.optim.Adam([x], lr=lr)
    for step in range(n_steps):
        opt.zero_grad()
        g = _grad_flat(model, x, y_label, device, create_graph=True)
        F.mse_loss(g, g_target.detach()).backward()
        opt.step()
        if (step + 1) % 500 == 0:
            with torch.no_grad():
                achieved = g.detach().abs().max().item()
                residual = (g.detach() - g_target).abs().max().item()
            print(f"    step {step+1:4d}: achieved L∞={achieved:.4f}  residual={residual:.4f}", flush=True)
    return x.detach()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', default='purchase')
    parser.add_argument('--model_name', default='mlp')
    parser.add_argument('--n_group_a', type=int, default=500,
                        help='Group A canaries — larger gradient norm, gets filtered by defense')
    parser.add_argument('--n_group_b', type=int, default=1000,
                        help='Group B canaries — smaller gradient norm, survives defense')
    parser.add_argument('--target_linf', type=float, default=50.0,
                        help='desired gradient L∞ for Group A (must be >> regular data max, typically ~5)')
    parser.add_argument('--label', type=int, default=0, help='label for Group A')
    parser.add_argument('--label_b', type=int, default=1, help='label for Group B')
    parser.add_argument('--n_steps', type=int, default=3000,
                        help='Adam steps per group')
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y, _ = load_data(args.data_name, n_df=None)
    input_dim = X.shape[1]
    out_dim = int(y.max().item()) + 1
    print(f"Dataset: {args.data_name}, input_dim={input_dim}, out_dim={out_dim}")

    model = Models[args.model_name](X.shape, out_dim=out_dim).to(device)
    xavier_init_model(model)
    model.eval()  # disable dropout for deterministic gradients

    # Group A target: random direction scaled to desired L∞
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    g_dir = torch.randn(n_params, device=device)
    g_A_target = g_dir / g_dir.abs().max() * args.target_linf

    linf_b_target = args.target_linf * args.n_group_a / args.n_group_b
    print(f"\nTargets: Group A L∞={args.target_linf:.1f}  Group B L∞={linf_b_target:.1f}  ratio={args.target_linf/linf_b_target:.1f}x")

    # Optimize x_A
    print(f"\nOptimizing x_A ({args.n_steps} steps)...")
    x_A = optimize_input(model, g_A_target, args.label, input_dim, device, args.n_steps, args.lr)

    # Compute actual achieved gradient for x_A, then set g_B to cancel it exactly
    g_A_achieved = _grad_flat(model, x_A, args.label, device)
    g_B_target = -(args.n_group_a / args.n_group_b) * g_A_achieved

    print(f"\nOptimizing x_B ({args.n_steps} steps)...")
    x_B = optimize_input(model, g_B_target, args.label_b, input_dim, device, args.n_steps, args.lr)

    # Verification
    g_B_achieved = _grad_flat(model, x_B, args.label_b, device)
    cancel_err = (args.n_group_a * g_A_achieved + args.n_group_b * g_B_achieved).abs().max().item()
    linf_a = g_A_achieved.abs().max().item()
    linf_b = g_B_achieved.abs().max().item()

    print(f"\nAchieved:")
    print(f"  Group A L∞ = {linf_a:.4f}  (target {args.target_linf:.1f})")
    print(f"  Group B L∞ = {linf_b:.4f}  (target {linf_b_target:.1f})")
    print(f"  Cancellation error (L∞ of n_A·g_A + n_B·g_B) = {cancel_err:.4e}")

    X_canary = torch.vstack([
        x_A.cpu().unsqueeze(0).expand(args.n_group_a, -1).clone(),
        x_B.cpu().unsqueeze(0).expand(args.n_group_b, -1).clone(),
    ])
    y_canary = torch.cat([
        torch.full((args.n_group_a,), args.label, dtype=torch.long),
        torch.full((args.n_group_b,), args.label_b, dtype=torch.long),
    ])

    out_path = output_dir / 'input_cancelling_canaries.pt'
    torch.save({
        'canaries': X_canary,
        'audit_labels': y_canary,
        'n_group_a': args.n_group_a,
        'n_group_b': args.n_group_b,
        'label_a': args.label,
        'label_b': args.label_b,
        'achieved_linf_a': linf_a,
        'achieved_linf_b': linf_b,
        'cancel_error': cancel_err,
        'input_dim': input_dim,
        'init_model': {k: v.cpu() for k, v in model.state_dict().items()},
    }, out_path)
    print(f"\nSaved {len(X_canary)} canaries ({args.n_group_a} A + {args.n_group_b} B) to {out_path}")


if __name__ == '__main__':
    main()
