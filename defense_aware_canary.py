"""
Defense-Aware Canary Generation: Fixed-Point Attack

We attempt to learn a canary that evades the L∞/L2 gradient-norm defense while
remaining auditable, via a fixed-point optimization at the initial model weights θ_0.

Objective:   L_canary = L_nn + λ * L_evade
  L_nn    = CE(f_θ(x), y)                          -- low loss = auditable
  L_evade = max(0, ||∇_θ ℓ(f_θ(x), y)||_norm - τ) -- hinge on gradient norm

τ is estimated as the top-k defense threshold on a clean holdout set.
Optimization is via FGSM (sign-gradient step) on x at fixed θ = θ_0.

Finding: for a wide range of λ, the resulting canary is still detected within the
first few training epochs.  The objective can push the initial gradient norm below τ,
but once training begins the canary's gradient norm grows as the model partially fits it,
making the evasion criterion impossible to maintain across epochs at a fixed-point solution.
"""

import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from torch.utils.data import DataLoader, Dataset

from opacus.accountants.utils import get_noise_multiplier

from models import Models
from utils.data import load_data
from utils.dpsgd import clip_and_accum_grads, DefenseConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def xavier_init_model(model):
    def init_weights(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)
    model.apply(init_weights)


class _IndexedDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i], i


# ---------------------------------------------------------------------------
# Threshold estimation
# ---------------------------------------------------------------------------

def estimate_threshold(model, X_holdout, y_holdout, defense_k, norm='l2', device='cuda:0'):
    """
    Estimate τ = defense threshold at θ_0 on clean holdout data.
    The top-k defense drops samples with gradient norms in the top-k per class.
    τ is taken as the global (1 - defense_k / n) quantile — the point below which
    a canary would not be filtered.
    """
    device_ = torch.device(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    norms = []
    for i in range(len(X_holdout)):
        x = X_holdout[i:i+1].to(device_)
        yt = y_holdout[i:i+1].to(device_)
        model.zero_grad()
        loss = criterion(model(x), yt)
        loss.backward()
        flat = torch.cat([p.grad.reshape(-1) for p in model.parameters() if p.grad is not None])
        if norm == 'l2':
            norms.append(flat.norm(p=2).item())
        else:
            norms.append(flat.abs().max().item())
    model.train()
    norms = np.array(norms)
    pct = 100.0 * (1.0 - defense_k / len(norms))
    tau = float(np.percentile(norms, pct))
    print(f"Threshold τ={tau:.6f} (defense_k={defense_k}, n={len(norms)}, "
          f"norm={norm}, p{pct:.1f} of holdout norms)")
    print(f"  Holdout norm stats: min={norms.min():.4f} p50={np.median(norms):.4f} "
          f"p90={np.percentile(norms,90):.4f} max={norms.max():.4f}")
    return tau


# ---------------------------------------------------------------------------
# Canary optimization
# ---------------------------------------------------------------------------

def _canary_grad_norm(model, x, y_t, criterion, norm):
    """
    Differentiable ||∇_θ L(f_θ(x), y)||_norm.
    create_graph=True allows backprop through the norm w.r.t. x.
    x must have requires_grad=True before calling.
    """
    loss = criterion(model(x.unsqueeze(0)), y_t)
    grads = torch.autograd.grad(
        loss, list(model.parameters()), create_graph=True, allow_unused=True
    )
    grads = [g for g in grads if g is not None]
    if norm == 'l2':
        return torch.sqrt(sum(g.pow(2).sum() for g in grads) + 1e-12)
    else:  # linf — subgradient at the max
        return torch.cat([g.reshape(-1) for g in grads]).abs().max()


def optimize_canary(model, x_init, y, tau, lam, norm, n_steps, step_size, device, verbose=False):
    """
    Minimize  L_nn + λ * max(0, ||∇_θ L||_norm - τ)  via FGSM on x at fixed θ.

    Returns (x_opt, final_loss_nn, final_grad_norm).
    """
    device_ = torch.device(device)
    criterion = nn.CrossEntropyLoss()
    y_t = torch.tensor([y], device=device_)
    x = x_init.clone().detach().to(device_)

    for step in range(n_steps):
        x = x.detach().requires_grad_(True)
        loss_nn = criterion(model(x.unsqueeze(0)), y_t)
        grad_norm = _canary_grad_norm(model, x, y_t, criterion, norm)
        loss_evade = torch.clamp(grad_norm - tau, min=0.0)
        total = loss_nn + lam * loss_evade
        grad_x = torch.autograd.grad(total, x)[0]
        with torch.no_grad():
            x = x - step_size * grad_x.sign()

        if verbose and step % max(1, n_steps // 5) == 0:
            print(f"    step {step:3d}: L_nn={loss_nn.item():.4f}  "
                  f"grad_norm={grad_norm.item():.4f}  L_evade={loss_evade.item():.4f}")

    # Final eval (no graph)
    x_final = x.detach()
    with torch.no_grad():
        loss_nn_final = criterion(model(x_final.unsqueeze(0)), y_t).item()

    model.zero_grad()
    x_eval = x_final.clone().requires_grad_(True)
    crit_eval = nn.CrossEntropyLoss()
    crit_eval(model(x_eval.unsqueeze(0)), y_t).backward()
    flat = torch.cat([p.grad.reshape(-1) for p in model.parameters() if p.grad is not None])
    if norm == 'l2':
        grad_norm_final = flat.norm(p=2).item()
    else:
        grad_norm_final = flat.abs().max().item()
    model.zero_grad()

    return x_final.cpu(), loss_nn_final, grad_norm_final


# ---------------------------------------------------------------------------
# Training with defense (to check if canary survives)
# ---------------------------------------------------------------------------

def train_with_defense(model_name, X, y, epsilon, delta, max_grad_norm, n_epochs,
                       lr, batch_size, init_state, out_dim, defense_k, norm, device, seed=0):
    """
    Train with DP-SGD + top-k gradient-norm defense.
    Canary must be the last sample in X / y (index len(X)-1).
    Returns (drop_epoch, score_history) where score_history is a list of
    (epoch, canary_grad_norm) pairs — recorded every epoch regardless of dropping.
    drop_epoch is -1 if the canary survived all epochs.
    """
    device_ = torch.device(device)
    model = Models[model_name](X.shape, out_dim=out_dim).to(device_)
    model.load_state_dict(deepcopy(init_state))
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    sample_rate = batch_size / len(X)
    noise_multiplier = get_noise_multiplier(
        target_epsilon=epsilon, target_delta=delta,
        sample_rate=sample_rate, epochs=n_epochs, accountant='rdp',
    )
    defense_cfg = DefenseConfig(score_fn='grad_norm_unclipped', score_norm=norm)

    n = len(X)
    canary_idx = n - 1
    scores = np.zeros(n, dtype=np.float32)
    drop_mask = np.zeros(n, dtype=np.int8)  # 0=active, 1=pending drop, 2=dropped

    dataset = _IndexedDataset(X, y)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0, drop_last=False,
    )

    drop_epoch = -1
    score_history = []  # (epoch, canary_grad_norm) — recorded even after drop

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        for curr_X, curr_y, global_indices in loader:
            curr_X = curr_X.to(device_)
            curr_y = curr_y.to(device_)
            global_indices = global_indices.to(device_)

            local_drop_mask = drop_mask[global_indices.cpu().numpy()]
            accum_grad, scores = clip_and_accum_grads(
                model, curr_X, curr_y, optimizer, criterion, max_grad_norm,
                block_size=batch_size, scores=scores, device=device,
                global_indices=global_indices, aug_mult=1, aug_fn=None,
                world_size=1, rank=0, batch_size=batch_size,
                drop_mask=local_drop_mask,
                defense_cfg=defense_cfg,
                defense_apply_ascent=False,
            )
            drop_mask[drop_mask == 1] = 2

            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name not in accum_grad:
                        continue
                    g = accum_grad[name].to(device_)
                    if noise_multiplier > 0:
                        g = g + noise_multiplier * max_grad_norm * torch.randn_like(g)
                    g.div_(float(len(curr_X)))
                    if param.grad is None:
                        param.grad = g
                    else:
                        param.grad.copy_(g)
            optimizer.step()
            optimizer.zero_grad()

        # Record canary score before defense clears it
        score_history.append((epoch, float(scores[canary_idx])))

        # End-of-epoch defense: drop top-k per class
        active_mask = torch.from_numpy(drop_mask == 0)
        for cls in torch.unique(y).tolist():
            cls_indices = ((y == int(cls)) & active_mask).nonzero(as_tuple=True)[0]
            if len(cls_indices) == 0:
                continue
            cls_scores = torch.from_numpy(scores[cls_indices.numpy()])
            k = min(defense_k, len(cls_scores))
            _, topk_local = torch.topk(cls_scores, k)
            topk_global = cls_indices[topk_local].numpy()
            drop_mask[topk_global] = 1
            if canary_idx in topk_global and drop_epoch == -1:
                drop_epoch = epoch
                print(f"    [epoch {epoch}] Canary detected and dropped (grad_norm={scores[canary_idx]:.4f})")

        scores.fill(0)

    return drop_epoch, score_history


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Defense-aware canary via fixed-point attack')
    parser.add_argument('--data_name',  type=str,   default='mnist')
    parser.add_argument('--model_name', type=str,   default='cnn')
    parser.add_argument('--out_dim',    type=int,   default=None)
    parser.add_argument('--n_holdout',  type=int,   default=500,
                        help='Samples to estimate τ on')
    # Canary optimization
    parser.add_argument('--lam',        type=float, nargs='+', default=[0.0, 0.1, 1.0, 10.0, 100.0],
                        help='Lambda sweep: weight on evasion penalty')
    parser.add_argument('--norm',       type=str,   default='l2', choices=['l2', 'linf'],
                        help='Gradient norm for both L_evade and defense scoring')
    parser.add_argument('--n_steps',    type=int,   default=50,
                        help='FGSM steps for canary optimization')
    parser.add_argument('--step_size',  type=float, default=0.01)
    parser.add_argument('--warm_steps', type=int,   default=0,
                        help='Train θ for this many SGD steps before optimizing the canary. '
                             '0 = fixed-point at θ_0 (original behaviour). '
                             'Nonzero = optimize against a warmed-up θ_t so the canary '
                             'evades the defense at the epoch it will actually fire, not just at init.')
    parser.add_argument('--init_mode',  type=str,   nargs='+', default=['blank', 'in_dist'],
                        choices=['blank', 'in_dist'],
                        help='Canary initialization strategy')
    parser.add_argument('--canary_label', type=int, default=0,
                        help='Label assigned to the canary')
    # Training check
    parser.add_argument('--check_training', action='store_true',
                        help='After optimization, run full DP-SGD training with defense to measure detection epoch')
    parser.add_argument('--n_runs',     type=int,   default=5,
                        help='Independent training runs per (lam, init_mode) when --check_training')
    parser.add_argument('--n_epochs',   type=int,   default=30)
    parser.add_argument('--lr',         type=float, default=3.0)
    parser.add_argument('--batch_size', type=int,   default=4000)
    parser.add_argument('--epsilon',    type=float, default=10.0)
    parser.add_argument('--delta',      type=float, default=1e-5)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--defense_k', type=int,   default=5)
    parser.add_argument('--n_df',      type=int,   default=None)
    # Misc
    parser.add_argument('--seed',   type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--output', type=str, default='defense_aware_canaries.pt')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    print(f"Loading {args.data_name}...")
    X, y, out_dim = load_data(args.data_name, n_df=args.n_df)
    if args.out_dim is not None:
        out_dim = args.out_dim
    X = X.float()
    y = y.long()
    print(f"  {len(X)} training samples, out_dim={out_dim}")

    # Fixed init weights shared across all experiments
    model = Models[args.model_name](X.shape, out_dim=out_dim).to(device)
    xavier_init_model(model)
    init_state = deepcopy(model.state_dict())

    # Estimate threshold τ on holdout
    print(f"\nEstimating defense threshold τ on {args.n_holdout} holdout samples...")
    X_holdout, y_holdout = X[:args.n_holdout], y[:args.n_holdout]
    tau = estimate_threshold(model, X_holdout, y_holdout, args.defense_k,
                             norm=args.norm, device=args.device)

    # Pick one in-distribution sample per class as init
    in_dist_inits = {}
    for cls in torch.unique(y).tolist():
        idx = (y == cls).nonzero(as_tuple=True)[0][0].item()
        in_dist_inits[cls] = X[idx].clone()

    results = []
    print(f"\n{'='*60}")
    print(f"Canary optimization sweep")
    print(f"  lam:       {args.lam}")
    print(f"  init_mode: {args.init_mode}")
    print(f"  norm:      {args.norm}")
    print(f"  n_steps:   {args.n_steps}  step_size: {args.step_size}")
    print(f"  τ = {tau:.6f}")
    print(f"{'='*60}")

    # Optionally warm up θ before optimizing (so canary evades at θ_t, not just θ_0)
    opt_model = deepcopy(model)
    if args.warm_steps > 0:
        print(f"\nWarming up θ for {args.warm_steps} SGD steps...")
        opt_model.train()
        warm_opt = torch.optim.SGD(opt_model.parameters(), lr=args.lr)
        criterion_warm = nn.CrossEntropyLoss()
        loader_warm = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X, y),
            batch_size=args.batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(args.seed),
        )
        steps_done = 0
        for xb, yb in loader_warm:
            warm_opt.zero_grad()
            criterion_warm(opt_model(xb.to(device)), yb.to(device)).backward()
            warm_opt.step()
            steps_done += 1
            if steps_done >= args.warm_steps:
                break
        tau_warm = estimate_threshold(opt_model, X_holdout, y_holdout, args.defense_k,
                                      norm=args.norm, device=args.device)
        print(f"  τ at warm θ: {tau_warm:.6f}  (was {tau:.6f} at θ_0)")
        tau = tau_warm

    opt_model.eval()
    for lam in args.lam:
        for init_mode in args.init_mode:
            print(f"\n--- λ={lam}  init={init_mode}  warm_steps={args.warm_steps} ---")

            if init_mode == 'blank':
                x_init = torch.zeros_like(X[0])
            else:
                x_init = in_dist_inits[args.canary_label].clone()

            x_opt, loss_nn, grad_norm = optimize_canary(
                opt_model, x_init, args.canary_label, tau, lam,
                args.norm, args.n_steps, args.step_size, args.device, verbose=True,
            )

            evaded = grad_norm < tau
            print(f"  Result: L_nn={loss_nn:.4f}  grad_norm={grad_norm:.6f}  "
                  f"τ={tau:.6f}  evaded_at_θ0={'YES' if evaded else 'NO'}")

            entry = dict(lam=lam, init_mode=init_mode, loss_nn=loss_nn,
                         grad_norm=grad_norm, tau=tau, evaded_at_theta0=evaded,
                         canary=x_opt, drop_epochs=[])

            if args.check_training:
                cls_mask = y == args.canary_label
                X_base = X[~cls_mask] if not cls_mask.all() else X.clone()
                y_base = y[~cls_mask] if not cls_mask.all() else y.clone()
                # Append canary as last sample
                X_with = torch.cat([X_base, x_opt.unsqueeze(0)], dim=0)
                y_with = torch.cat([y_base,
                                    torch.tensor([args.canary_label], dtype=torch.long)], dim=0)

                drop_epochs = []
                all_score_histories = []
                for run in range(args.n_runs):
                    print(f"  Training run {run+1}/{args.n_runs}...", end=' ', flush=True)
                    t0 = time.time()
                    drop_ep, score_hist = train_with_defense(
                        args.model_name, X_with, y_with,
                        args.epsilon, args.delta, args.max_grad_norm,
                        args.n_epochs, args.lr, args.batch_size,
                        init_state, out_dim, args.defense_k, args.norm,
                        args.device, seed=args.seed + run,
                    )
                    drop_epochs.append(drop_ep)
                    all_score_histories.append(score_hist)
                    status = f"dropped ep={drop_ep}" if drop_ep != -1 else "survived"
                    print(f"{status}  ({time.time()-t0:.1f}s)")

                entry['drop_epochs'] = drop_epochs
                entry['score_histories'] = all_score_histories
                survived = sum(1 for e in drop_epochs if e == -1)
                avg_drop = np.mean([e for e in drop_epochs if e != -1]) if any(e != -1 for e in drop_epochs) else float('nan')
                print(f"  Summary: survived {survived}/{args.n_runs} runs, "
                      f"avg detection epoch={avg_drop:.1f}")

                # Print mean canary score trajectory across runs (up to first drop)
                max_ep = max(len(h) for h in all_score_histories)
                print(f"  Canary grad_norm trajectory (mean across runs)  [τ={tau:.4f}]:")
                for ep in range(min(max_ep, 10)):  # first 10 epochs
                    ep_scores = [h[ep][1] for h in all_score_histories if ep < len(h)]
                    bar = '↑ ABOVE τ' if np.mean(ep_scores) > tau else '  below τ'
                    print(f"    ep {ep:2d}: {np.mean(ep_scores):.4f} ± {np.std(ep_scores):.4f}  {bar}")

            results.append(entry)

    torch.save({'results': results, 'args': vars(args), 'tau': tau}, args.output)
    print(f"\nSaved {len(results)} results to {args.output}")

    # Summary table
    print(f"\n{'λ':>8}  {'init':>8}  {'L_nn':>8}  {'grad_norm':>10}  {'evaded@θ0':>10}  {'drop_epochs'}")
    print('-' * 70)
    for r in results:
        de_str = str(r['drop_epochs']) if r['drop_epochs'] else '(not checked)'
        print(f"{r['lam']:>8.1f}  {r['init_mode']:>8}  {r['loss_nn']:>8.4f}  "
              f"{r['grad_norm']:>10.6f}  {str(r['evaded_at_theta0']):>10}  {de_str}")


if __name__ == '__main__':
    main()
