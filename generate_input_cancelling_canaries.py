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
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader

from models import Models
from utils.data import load_data
from parallel_audit_multi_canary import train_model_multi_canary


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


def _check_canary_grads_at_epoch(model, random_dense, alpha, beta, label_a, label_b, X_ref, y_ref, device):
    """Quick check of Group A and B gradient norms at current epoch (dense canary version)."""
    input_dim = X_ref.shape[1]
    model.eval()

    # Group A: +random_dense * alpha
    x_a = (random_dense * alpha).unsqueeze(0).to(device)
    y_a = torch.tensor([label_a], device=device)
    model.zero_grad()
    F.cross_entropy(model(x_a), y_a).backward()
    flat_a = torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None])
    linf_a = float(flat_a.abs().max().item())
    model.zero_grad()

    # Group B: -random_dense * beta (opposite direction for cancellation)
    x_b = (-random_dense * beta).unsqueeze(0).to(device)
    y_b = torch.tensor([label_b], device=device)
    model.zero_grad()
    F.cross_entropy(model(x_b), y_b).backward()
    flat_b = torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None])
    linf_b = float(flat_b.abs().max().item())
    model.zero_grad()

    print(f"  GroupA (+α={alpha:.2f}): L∞={linf_a:.6f}")
    print(f"  GroupB (-β={beta:.2f}): L∞={linf_b:.6f}  (ratio B/A: {linf_b/linf_a if linf_a > 0 else float('inf'):.2f}x)")


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
    parser.add_argument('--n_epochs', type=int, default=100,
                        help='epochs of training WITH canaries for calibration (match actual training)')
    parser.add_argument('--lr', type=float, default=0.5,
                        help='learning rate')
    parser.add_argument('--batch_size', type=int, default=12143,
                        help='batch size for Poisson sampling')
    parser.add_argument('--defense', action='store_true', default=True,
                        help='enable gradient filtering defense during calibration')
    parser.add_argument('--defense_k', type=int, default=5,
                        help='number of top samples to filter per epoch')
    parser.add_argument('--defense_score_fn', type=str, default='grad_norm_unclipped',
                        help='defense scoring function')
    parser.add_argument('--defense_score_norm', type=str, default='linf',
                        help='defense score norm type')
    parser.add_argument('--sampling', type=str, default='poisson',
                        help='sampling strategy (poisson or full)')
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

    print(f"Training briefly (1 epoch) to identify hot input dimension …")
    train_briefly(model, X, y, device, 1, args.lr, min(args.batch_size, len(X)))

    hot_dim = find_hot_input_dim(model, X, y, device)

    beta = args.beta if args.beta is not None else (args.n_group_a * args.alpha / args.n_group_b)
    print(f"\nInput-Space Cancelling Canary Design:")
    print(f"  Random unit vector seed={args.seed}, input_dim={input_dim}")
    print(f"  Group A: {args.n_group_a} × (+random_dense × {args.alpha:.4f}), label={args.label}  ← filtered by defense (larger per-sample norm α)")
    print(f"  Group B: {args.n_group_b} × (-random_dense × {beta:.4f}),  label={args.label_b}  ← survives defense (smaller per-sample norm β)")
    print(f"  Cancellation: {args.n_group_a}×{args.alpha:.4f} = {args.n_group_b}×{beta:.4f}  ({args.n_group_a*args.alpha:.4f} vs {args.n_group_b*beta:.4f})")
    print(f"  α/β = {args.alpha/beta:.1f}x  (both >> regular data norms ~1.6)")

    # Dense input-space cancelling canaries.
    #
    # Both groups use the same random unit vector but with opposite sign,
    # so their total input contribution sums to zero:
    #   n_group_a * (+random_dense * alpha) + n_group_b * (-random_dense * beta) = 0
    #   => n_group_a * alpha = n_group_b * beta  (cancellation constraint)
    #   => beta = n_group_a * alpha / n_group_b  (default if not passed)
    #
    # alpha > beta (since n_group_b > n_group_a), so Group A has larger per-sample norm
    # and gets filtered by the defense. Group B (smaller norm) survives and provides audit signal.
    # Both alpha and beta >> normal sample norms.

    torch.manual_seed(args.seed)
    random_dense = torch.randn(input_dim)
    random_dense = random_dense / random_dense.norm(p=2)

    # Group A: +direction, larger per-sample norm, gets filtered by defense
    x_a = random_dense * args.alpha
    X_a = x_a.unsqueeze(0).expand(args.n_group_a, -1).clone()
    y_a = torch.full((args.n_group_a,), args.label, dtype=torch.long)

    # Group B: -direction, smaller per-sample norm (beta = n_a*alpha/n_b < alpha), survives defense
    x_b = -random_dense * beta
    X_b = x_b.unsqueeze(0).expand(args.n_group_b, -1).clone()
    y_b = torch.full((args.n_group_b,), args.label_b, dtype=torch.long)

    X_canary = torch.vstack([X_a, X_b])
    y_canary = torch.cat([y_a, y_b])

    # Calibrate using the exact same training code as the actual audit
    print(f"\nCalibrating with canaries using train_model_multi_canary (exact audit setup) …")
    X_with_canaries = torch.cat([X, X_canary], dim=0)
    y_with_canaries = torch.cat([y, y_canary], dim=0)

    # Canary indices: last len(X_canary) samples
    canary_indices = np.arange(len(X), len(X_with_canaries), dtype=np.int64)

    model_calibrate = Models[args.model_name](X_with_canaries.shape, out_dim=out_dim).to(device)
    xavier_init_model(model_calibrate)

    # Call the exact same training function used in the audit
    model_calibrate, drop_mask, defense_stats = train_model_multi_canary(
        model_name=args.model_name,
        X=X_with_canaries,
        y=y_with_canaries,
        epsilon=None,  # non-private
        delta=None,
        max_grad_norm=None,  # non-private
        n_epochs=int(args.n_epochs),
        lr=float(args.lr),
        block_size=int(args.batch_size),
        batch_size=int(args.batch_size),
        init_model=model_calibrate,
        out_dim=out_dim,
        aug_mult=1,
        defense=bool(args.defense),
        defense_k=int(args.defense_k),
        defense_apply_ascent=False,
        defense_filter_every=1,
        defense_score_fn=str(args.defense_score_fn),
        defense_score_norm=str(args.defense_score_norm),
        defense_global_filter=False,
        device=str(device),
        generator=None,
        dl_generator=None,
        num_workers=0,
        persistent_workers=False,
        canary_indices=canary_indices,
        sampling=args.sampling,
        is_gradient_space_canary=False,
        global_idx_to_grad=None,
        rank=0,
    )

    print(f"\nCalibration complete. Canary drop stats: {defense_stats}")

    # Check final gradient norms on trained model
    print(f"\nFinal canary gradient norms on trained model:")
    _check_canary_grads_at_epoch(model_calibrate, random_dense, args.alpha, beta,
                                args.label, args.label_b,
                                X, y, device)

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
