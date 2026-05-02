"""
Generate input-space cancelling canaries for Purchase/MLP.

For an MLP the per-sample gradient at the first layer is  δ₁ ⊗ x,  so the input
vector x directly determines the gradient direction in that layer.  Cancellation
can therefore be designed in input space:

  Group A : n_group_a canaries, input[hot_dim] = +alpha, label = label_a  (low  L∞ gradient → evades defence)
  Group B : n_group_b canaries, input[hot_dim] = +beta,  label = label_b  (high L∞ gradient → removed by defence)
  Constraint: n_group_a * alpha = n_group_b * beta                          (approximate gradient cancellation)

WHY POSITIVE BETA: using −beta at hot_dim kills ReLU neurons (negative pre-activation → zero gradient),
so the full-gradient L∞ seen by the defence is comparable to regular data and group B goes undetected.
Using +beta keeps all neurons alive; gradient magnitude at column hot_dim scales with beta, making
group B clearly anomalous to the L∞ defence.

WHY DIFFERENT LABELS: same input direction with same label produces additive (not cancelling) gradients.
Different labels give approximately opposite error signals δ_A ≈ −δ_B (exact at K=2, approximate at K=100),
so the net first-layer gradient contribution is ≈ 0 when n_A·alpha = n_B·beta.

Without defence:  A + B in training → net gradient ≈ 0 → no MIA gap.
With defence:     Group B (large L∞) removed → A alone memorised → gap appears.

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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


def measure_grad_norm_distribution(model, X, y, device, n_samples=2000, batch_size=256):
    """Measure per-sample gradient L2 and L∞ norm distribution on regular training data."""
    model.train()
    l2_norms, linf_norms = [], []

    n = min(n_samples, len(X))
    loader = DataLoader(
        TensorDataset(X[:n], y[:n]),
        batch_size=batch_size, shuffle=False,
    )

    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        for i in range(len(X_b)):
            model.zero_grad()
            loss = F.cross_entropy(model(X_b[i:i+1]), y_b[i:i+1])
            loss.backward()
            flat = torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None])
            l2_norms.append(float(flat.norm(2).item()))
            linf_norms.append(float(flat.abs().max().item()))

    l2 = np.array(l2_norms)
    linf = np.array(linf_norms)

    print(f"\n=== Per-sample gradient norm distribution (n={n}) ===")
    for p in [50, 75, 90, 95, 99]:
        print(f"  p{p:2d}:  L2={np.percentile(l2, p):.4f}  L∞={np.percentile(linf, p):.4f}")
    print(f"  max:  L2={l2.max():.4f}  L∞={linf.max():.4f}")
    print(f"===================================================\n")
    return l2, linf


def check_canary_calibration(model, hot_dim, beta, label_b, X_ref, y_ref, device,
                              n_group_a, alpha, defense_k=5, n_regular=500):
    """Measure group B's actual gradient L∞ vs regular class-{label_b} data.

    For a multi-layer MLP, group B's gradient L∞ is dominated by the first layer
    (sparse 1-hot input), while regular data's L∞ is dominated by later layers
    (dense activations). This function measures the true relationship so you can
    calibrate beta correctly.

    Prints: group B's actual L∞, comparison to class-label_b regular data, and
    suggested n_group_b if current beta is insufficient.
    """
    input_dim = X_ref.shape[1]
    model.train()

    # Group B canary: x = beta * e_hot_dim (the actual canary input, not scaled unit)
    x_b = torch.zeros(1, input_dim, device=device)
    x_b[0, hot_dim] = beta
    y_b = torch.tensor([label_b], device=device)
    model.zero_grad()
    F.cross_entropy(model(x_b), y_b).backward()
    flat_b_actual = torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None])
    linf_b = float(flat_b_actual.abs().max().item())
    l2_b = float(flat_b_actual.norm(2).item())
    model.zero_grad()

    # Also measure unit input to expose how much scaling actually helps
    x_unit = torch.zeros(1, input_dim, device=device)
    x_unit[0, hot_dim] = 1.0
    model.zero_grad()
    F.cross_entropy(model(x_unit), y_b).backward()
    flat_unit = torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None])
    linf_unit = float(flat_unit.abs().max().item())
    l2_unit = float(flat_unit.norm(2).item())
    model.zero_grad()

    # Regular class-{label_b} data
    cls_idx = (y_ref == label_b).nonzero(as_tuple=True)[0]
    n_samp = min(n_regular, len(cls_idx))
    X_cls = X_ref[cls_idx[:n_samp]].to(device)
    y_cls = y_ref[cls_idx[:n_samp]].to(device)
    linf_reg = []
    for i in range(n_samp):
        model.zero_grad()
        F.cross_entropy(model(X_cls[i:i+1]), y_cls[i:i+1]).backward()
        flat = torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None])
        linf_reg.append(float(flat.abs().max().item()))
    model.zero_grad()
    linf_reg = np.array(linf_reg)

    p90 = np.percentile(linf_reg, 90)
    rank = int(np.sum(linf_reg > linf_b))  # how many regular samples beat group B

    print(f"\n=== Group B canary calibration (hot_dim={hot_dim}, β={beta:.2f}) ===")
    print(f"  Unit-input gradient:  L∞={linf_unit:.6f}  L2={l2_unit:.6f}")
    print(f"  Group B L∞ (β×unit): {linf_b:.6f}  L2={l2_b:.6f}")
    print(f"  Class-{label_b} regular:  p50={np.percentile(linf_reg,50):.6f}  p90={p90:.6f}  max={linf_reg.max():.6f}")
    print(f"  Group B beats {n_samp - rank}/{n_samp} regular class-{label_b} samples")
    if rank < defense_k:
        print(f"  ✓  β={beta:.2f} is sufficient — group B will be in top-{defense_k} and filtered")
    else:
        needed_beta = p90 / linf_unit if linf_unit > 0 else float('inf')
        needed_n_b = max(1, int(n_group_a * alpha / needed_beta) + 1)
        print(f"  ✗  β={beta:.2f} insufficient — group B is NOT in top-{defense_k}")
        print(f"     Need β > {needed_beta:.1f}  →  set n_group_b ≤ {needed_n_b}  (= {n_group_a}×{alpha}/{needed_beta:.0f})")
    print(f"=================================================================\n")


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
                        help='audit label for group A canaries')
    parser.add_argument('--label_b', type=int, default=1,
                        help='audit label for group B canaries (should differ from --label to get opposing error signal)')
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

    measure_grad_norm_distribution(model, X, y, device)

    print(f"Training briefly ({args.n_epochs} epochs) to identify hot input dimension …")
    train_briefly(model, X, y, device, args.n_epochs, args.lr, args.batch_size)

    hot_dim = find_hot_input_dim(model, X, y, device)

    beta = args.beta if args.beta is not None else (args.n_group_a * args.alpha / args.n_group_b)
    print(f"\nGroup A: {args.n_group_a} canaries, input[{hot_dim}] = +{args.alpha:.4f}, label={args.label}")
    print(f"Group B: {args.n_group_b} canaries, input[{hot_dim}] = +{beta:.4f},  label={args.label_b}  ← positive keeps ReLU alive")
    print(f"Cancellation check: {args.n_group_a} * {args.alpha} = {args.n_group_b} * {beta:.4f}  "
          f"→ {args.n_group_a * args.alpha:.4f} vs {args.n_group_b * beta:.4f}")
    print(f"Note: cancellation is approximate (δ_A ≈ −δ_B for different labels, exact only at K=2)")

    check_canary_calibration(
        model, hot_dim, beta, args.label_b, X, y, device,
        n_group_a=args.n_group_a, alpha=args.alpha,
        defense_k=5,
    )

    # Build group A: 1-hot in input space at hot_dim with value +alpha
    x_a = torch.zeros(input_dim)
    x_a[hot_dim] = args.alpha
    X_a = x_a.unsqueeze(0).expand(args.n_group_a, -1).clone()
    y_a = torch.full((args.n_group_a,), args.label, dtype=torch.long)

    # Build group B: 1-hot in input space at hot_dim with value +beta (positive — keeps ReLU neurons alive
    # so the first-layer gradient magnitude ∝ beta and the L∞ defence can detect it)
    x_b = torch.zeros(input_dim)
    x_b[hot_dim] = beta
    X_b = x_b.unsqueeze(0).expand(args.n_group_b, -1).clone()
    y_b = torch.full((args.n_group_b,), args.label_b, dtype=torch.long)

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
        'label_a': args.label,
        'label_b': args.label_b,
        'input_dim': input_dim,
    }, out_path)
    print(f"\nSaved {len(X_canary)} canaries to {out_path}")
    print(f"  shape: {tuple(X_canary.shape)}, labels: {y_canary.unique().tolist()}")


if __name__ == '__main__':
    main()
