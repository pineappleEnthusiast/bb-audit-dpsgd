"""
Mislabeled Canary: Trajectory-Aware Defense Evasion

A mislabeled canary (real image, wrong label) optimized via FGSM against the
full training trajectory, not just the initial model.

Objective (n_steps of FGSM on x at fixed checkpoints {θ_t}):
    min_x  CE(f_θ₀(x), y_wrong)
         + λ * Σ_t max(0, ||∇_θ L(f_θ_t(x), y_wrong)||_linf − τ_t)

Phase 1 — collect_trajectory(): trains a clean model (no canary) and saves
  checkpoints + τ_t at epochs specified by --checkpoint_epochs.

Phase 2 — optimize_canary_trajectory(): FGSM on x, penalizing gradient norm
  above τ_t at every checkpoint. Lower initial loss = faster memorization;
  penalty keeps the canary below the defense threshold throughout training.

Phase 3 — train_with_defense(): runs the actual DP-SGD audit to measure the
  detection epoch for the optimized canary.
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


def _grad_norm(model, x, y_t, criterion, norm):
    """Per-sample gradient norm of CE(f_θ(x), y_t) w.r.t. θ. x must already have batch dim."""
    param_grads = torch.autograd.grad(
        criterion(model(x), y_t), list(model.parameters()),
        create_graph=True, allow_unused=True,
    )
    flat = torch.cat([g.reshape(-1) for g in param_grads if g is not None])
    return flat.abs().max() if norm == 'linf' else flat.norm(p=2)


# ---------------------------------------------------------------------------
# Phase 1: collect trajectory checkpoints
# ---------------------------------------------------------------------------

def collect_trajectory(model_name, X, y, epsilon, delta, max_grad_norm, n_epochs,
                       lr, batch_size, out_dim, defense_k, norm,
                       checkpoint_epochs, X_holdout, y_holdout, device, seed=0):
    """
    Train a clean DP-SGD model (no canary) and collect (state_dict, τ_t) at
    each epoch in checkpoint_epochs.
    Returns list of (epoch, state_dict, tau_t).
    """
    device_ = torch.device(device)
    model = Models[model_name](X.shape, out_dim=out_dim).to(device_)
    xavier_init_model(model)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    sample_rate = batch_size / len(X)
    noise_multiplier = get_noise_multiplier(
        target_epsilon=epsilon, target_delta=delta,
        sample_rate=sample_rate, epochs=n_epochs, accountant='rdp',
    )

    defense_cfg = DefenseConfig(score_fn='grad_norm_unclipped', score_norm=norm)
    dataset = _IndexedDataset(X, y)
    n = len(dataset)
    scores    = np.zeros(n, dtype=np.float32)
    drop_mask = np.zeros(n, dtype=np.int8)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed),
                        num_workers=0, drop_last=False)

    checkpoint_set = set(checkpoint_epochs)
    checkpoints = []

    # Always capture θ₀ before any training
    if -1 in checkpoint_set or 0 in checkpoint_set:
        tau_0 = _estimate_tau(model, X_holdout, y_holdout, defense_k, norm, device)
        checkpoints.append((0, deepcopy(model.state_dict()), tau_0))
        print(f"  checkpoint ep=0  τ={tau_0:.6f}")

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        for curr_X, curr_y, global_indices in loader:
            curr_X = curr_X.to(device_)
            curr_y = curr_y.to(device_)
            global_indices = global_indices.to(device_)

            local_dm = drop_mask[global_indices.cpu().numpy()]
            accum_grad, scores = clip_and_accum_grads(
                model, curr_X, curr_y, optimizer, criterion, max_grad_norm,
                block_size=batch_size, scores=scores, device=device,
                global_indices=global_indices, aug_mult=1, aug_fn=None,
                world_size=1, rank=0, batch_size=batch_size,
                drop_mask=local_dm, defense_cfg=deepcopy(defense_cfg),
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
                    param.grad = g if param.grad is None else param.grad.copy_(g)
            optimizer.step()
            optimizer.zero_grad()

        # End-of-epoch defense
        active_mask = torch.from_numpy(drop_mask == 0)
        for cls in torch.unique(y).tolist():
            cls_idx = ((y == int(cls)) & active_mask).nonzero(as_tuple=True)[0]
            if len(cls_idx) == 0:
                continue
            cls_sc = torch.from_numpy(scores[cls_idx.numpy()])
            _, topk = torch.topk(cls_sc, min(defense_k, len(cls_sc)))
            drop_mask[cls_idx[topk].numpy()] = 1
        scores.fill(0)

        ep1 = epoch + 1
        if ep1 in checkpoint_set:
            tau_t = _estimate_tau(model, X_holdout, y_holdout, defense_k, norm, device)
            checkpoints.append((ep1, deepcopy(model.state_dict()), tau_t))
            print(f"  checkpoint ep={ep1}  τ={tau_t:.6f}  "
                  f"active={int((drop_mask == 0).sum())}/{n}")

    return checkpoints


def _estimate_tau(model, X_holdout, y_holdout, defense_k, norm, device):
    """τ = (1 - k/n) quantile of per-sample gradient norms on holdout at current θ."""
    device_ = torch.device(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    norms = []
    with torch.no_grad():
        pass  # just to be safe
    for i in range(len(X_holdout)):
        x = X_holdout[i:i+1].to(device_)
        yt = y_holdout[i:i+1].to(device_)
        model.zero_grad()
        loss = criterion(model(x), yt)
        loss.backward()
        flat = torch.cat([p.grad.reshape(-1) for p in model.parameters()
                          if p.grad is not None])
        norms.append(flat.abs().max().item() if norm == 'linf' else flat.norm(p=2).item())
        model.zero_grad()
    model.train()
    norms = np.array(norms)
    pct = 100.0 * (1.0 - defense_k / len(norms))
    return float(np.percentile(norms, pct))


# ---------------------------------------------------------------------------
# Phase 2: trajectory-aware canary optimization
# ---------------------------------------------------------------------------

def optimize_canary_trajectory(checkpoints, x_init, wrong_label, norm,
                                n_steps, step_size, lam, device, verbose=True):
    """
    FGSM minimizing:
        CE(f_θ₀(x), y_wrong) + λ * Σ_t max(0, ||∇_θ L(f_θ_t(x), y_wrong)||_linf − τ_t)

    checkpoints: list of (epoch, state_dict, tau_t, model) from collect_trajectory().
    n_steps=0 returns the unoptimized init as a baseline.
    Returns (x_opt, final_loss_nn, {epoch: grad_norm}).
    """
    device_ = torch.device(device)
    criterion = nn.CrossEntropyLoss()
    y_t = torch.tensor([wrong_label], device=device_)

    ep0, sd0, tau0, _ = checkpoints[0]

    x = x_init.clone().detach().to(device_)

    for step in range(n_steps):
        x = x.detach().requires_grad_(True)
        x_b = x.unsqueeze(0)

        total_loss = torch.tensor(0.0, device=device_)
        grad_norms_this_step = {}

        for ep, state_dict, tau_t, ckpt_model in checkpoints:
            ckpt_model.eval()
            loss_t = criterion(ckpt_model(x_b), y_t)

            if ep == ep0:
                # θ₀ term: primary auditability objective
                total_loss = total_loss + loss_t

            # Evasion penalty at this checkpoint
            gn = _grad_norm(ckpt_model, x_b, y_t, criterion, norm)
            penalty = torch.clamp(gn - tau_t, min=0.0)
            total_loss = total_loss + lam * penalty
            grad_norms_this_step[ep] = gn.item()

        grad_x = torch.autograd.grad(total_loss, x)[0]
        x = x.detach() - step_size * grad_x.sign()

        # Zero out all checkpoint model grads
        for _, _, _, m in checkpoints:
            m.zero_grad()

        if verbose and step % max(1, n_steps // 5) == 0:
            penalty_str = '  '.join(
                f"ep{ep}:{gn:.4f}(τ={tau_t:.4f})"
                for (ep, _, tau_t, _), gn in zip(checkpoints, grad_norms_this_step.values())
            )
            print(f"    step {step:3d}: {penalty_str}")

    x_final = x.detach().cpu()

    # Final evaluation
    x_eval = x_final.to(device_).unsqueeze(0)
    final_norms = {}
    for ep, state_dict, tau_t, ckpt_model in checkpoints:
        ckpt_model.eval()
        ckpt_model.zero_grad()
        loss_f = criterion(ckpt_model(x_eval.requires_grad_(True) if ep == ep0 else x_eval), y_t)
        if ep == ep0:
            loss_nn_final = loss_f.item()
        pg = torch.autograd.grad(criterion(ckpt_model(x_eval.clone().requires_grad_(True)), y_t),
                                 list(ckpt_model.parameters()), allow_unused=True)
        flat = torch.cat([g.reshape(-1) for g in pg if g is not None])
        final_norms[ep] = flat.abs().max().item() if norm == 'linf' else flat.norm(p=2).item()
        ckpt_model.zero_grad()

    return x_final, loss_nn_final, final_norms


# ---------------------------------------------------------------------------
# Phase 3: training check
# ---------------------------------------------------------------------------

def train_with_defense(model_name, X, y, epsilon, delta, max_grad_norm, n_epochs,
                       lr, batch_size, init_state, out_dim, defense_k, norm, device, seed=0):
    """
    DP-SGD + top-k gradient-norm defense. Canary is the last sample (index len(X)-1).
    Returns (drop_epoch, score_history).  drop_epoch=-1 means survived all epochs.
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
    scores    = np.zeros(n, dtype=np.float32)
    drop_mask = np.zeros(n, dtype=np.int8)

    dataset = _IndexedDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed),
                        num_workers=0, drop_last=False)

    drop_epoch   = -1
    score_history = []

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        for curr_X, curr_y, global_indices in loader:
            curr_X = curr_X.to(device_)
            curr_y = curr_y.to(device_)
            global_indices = global_indices.to(device_)

            local_dm = drop_mask[global_indices.cpu().numpy()]
            accum_grad, scores = clip_and_accum_grads(
                model, curr_X, curr_y, optimizer, criterion, max_grad_norm,
                block_size=batch_size, scores=scores, device=device,
                global_indices=global_indices, aug_mult=1, aug_fn=None,
                world_size=1, rank=0, batch_size=batch_size,
                drop_mask=local_dm, defense_cfg=defense_cfg,
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
                    param.grad = g if param.grad is None else param.grad.copy_(g)
            optimizer.step()
            optimizer.zero_grad()

        score_history.append((epoch, float(scores[canary_idx])))

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
                print(f"    [epoch {epoch}] detected  grad_norm={scores[canary_idx]:.4f}")

        scores.fill(0)

    return drop_epoch, score_history


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Trajectory-aware mislabeled canary')
    parser.add_argument('--data_name',   type=str, default='mnist')
    parser.add_argument('--model_name',  type=str, default='cnn')
    parser.add_argument('--out_dim',     type=int, default=None)
    parser.add_argument('--n_holdout',   type=int, default=500)
    parser.add_argument('--norm',        type=str, default='linf', choices=['l2', 'linf'])
    parser.add_argument('--source_classes', type=int, nargs='+', default=None)
    parser.add_argument('--canary_label',type=int, default=0)
    # Trajectory collection
    parser.add_argument('--checkpoint_epochs', type=int, nargs='+',
                        default=[0, 5, 10, 20, 50, 100],
                        help='Epochs at which to save model checkpoints for optimization')
    # Canary optimization
    parser.add_argument('--lam',         type=float, nargs='+', default=[1.0],
                        help='Penalty weight(s) on gradient-norm evasion term. '
                             'One result per λ per source class.')
    parser.add_argument('--n_steps',     type=int,   nargs='+', default=[0, 100],
                        help='FGSM step counts to sweep. 0 = unoptimized baseline.')
    parser.add_argument('--step_size',   type=float, nargs='+', default=[0.01],
                        help='FGSM step size(s) to sweep.')
    # Training check
    parser.add_argument('--n_runs',      type=int,   default=10)
    parser.add_argument('--n_epochs',    type=int,   default=100)
    parser.add_argument('--lr',          type=float, default=3.0)
    parser.add_argument('--batch_size',  type=int,   default=4000)
    parser.add_argument('--epsilon',     type=float, default=10.0)
    parser.add_argument('--delta',       type=float, default=1e-5)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--defense_k',   type=int,   default=5)
    parser.add_argument('--n_df',        type=int,   default=None)
    parser.add_argument('--seed',        type=int,   default=0)
    parser.add_argument('--device',      type=str,   default='cuda:0')
    parser.add_argument('--output',      type=str,   default='mislabeled_canaries.pt')
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
    print(f"  {len(X)} samples, out_dim={out_dim}")

    # Fixed init shared across all experiments
    ref_model = Models[args.model_name](X.shape, out_dim=out_dim).to(device)
    xavier_init_model(ref_model)
    init_state = deepcopy(ref_model.state_dict())

    X_holdout, y_holdout = X[:args.n_holdout], y[:args.n_holdout]

    # ------------------------------------------------------------------ #
    # Phase 1: collect trajectory checkpoints on clean data               #
    # ------------------------------------------------------------------ #
    ckpt_epochs = sorted(set(args.checkpoint_epochs))
    max_ckpt_epoch = max(ckpt_epochs)
    print(f"\n{'='*60}")
    print(f"Phase 1: collecting trajectory checkpoints at epochs {ckpt_epochs}")
    print(f"{'='*60}")
    raw_checkpoints = collect_trajectory(
        model_name=args.model_name, X=X, y=y,
        epsilon=args.epsilon, delta=args.delta,
        max_grad_norm=args.max_grad_norm,
        n_epochs=max_ckpt_epoch, lr=args.lr, batch_size=args.batch_size,
        out_dim=out_dim, defense_k=args.defense_k, norm=args.norm,
        checkpoint_epochs=ckpt_epochs,
        X_holdout=X_holdout, y_holdout=y_holdout,
        device=args.device, seed=args.seed,
    )
    # Build model objects for each checkpoint (reuse architecture, swap weights)
    checkpoints_with_models = []
    for ep, sd, tau_t in raw_checkpoints:
        m = Models[args.model_name](X.shape, out_dim=out_dim).to(device)
        m.load_state_dict(sd)
        m.eval()
        checkpoints_with_models.append((ep, sd, tau_t, m))
    print(f"Collected {len(checkpoints_with_models)} checkpoints.")

    # ------------------------------------------------------------------ #
    # Phase 2+3: optimize canary, then check detection for each src class #
    # ------------------------------------------------------------------ #
    all_classes   = torch.unique(y).tolist()
    source_classes = args.source_classes or [c for c in all_classes if c != args.canary_label]
    class_exemplars = {
        cls: X[(y == cls).nonzero(as_tuple=True)[0][0].item()].clone()
        for cls in all_classes
    }

    results = []
    print(f"\n{'='*60}")
    print(f"Phase 2+3: canary optimization + detection check")
    print(f"  source_classes={source_classes}  canary_label={args.canary_label}")
    print(f"  λ sweep: {args.lam}  n_steps={args.n_steps}  step_size={args.step_size}")
    print(f"{'='*60}")

    for src_cls in source_classes:
        x_init = class_exemplars[src_cls].clone()

        for n_steps in args.n_steps:
            for step_size in args.step_size:
                for lam in (args.lam if n_steps > 0 else [0.0]):
                    label = (f"src={src_cls} → {args.canary_label}  "
                             f"n_steps={n_steps}  step_size={step_size}  λ={lam}")
                    print(f"\n--- {label} ---")

                    x_canary, loss_nn_opt, final_norms = optimize_canary_trajectory(
                        checkpoints_with_models, x_init, args.canary_label,
                        args.norm, n_steps, step_size, lam, args.device,
                        verbose=(n_steps > 0),
                    )

                    print(f"  Final: L_nn={loss_nn_opt:.4f}")
                    for ep, _, tau_t, _ in checkpoints_with_models:
                        gn = final_norms[ep]
                        status = 'ABOVE τ ✗' if gn >= tau_t else 'below τ ✓'
                        print(f"    ep={ep:3d}: grad_norm={gn:.6f}  τ={tau_t:.6f}  {status}")

                    X_with = torch.cat([X[:-1], x_canary.unsqueeze(0)], dim=0)
                    y_with = torch.cat([y[:-1], torch.tensor([args.canary_label],
                                        dtype=torch.long)], dim=0)

                    drop_epochs, all_score_histories = [], []
                    for run in range(args.n_runs):
                        print(f"  run {run+1}/{args.n_runs}...", end=' ', flush=True)
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

                    survived = sum(1 for e in drop_epochs if e == -1)
                    detected = [e for e in drop_epochs if e != -1]
                    avg_drop = np.mean(detected) if detected else float('nan')
                    print(f"  Summary: survived {survived}/{args.n_runs}  "
                          f"avg_detection_epoch={avg_drop:.1f}  drop_epochs={drop_epochs}")

                    results.append(dict(
                        src_cls=src_cls, canary_label=args.canary_label,
                        n_steps=n_steps, step_size=step_size, lam=lam,
                        canary=x_canary.cpu(), loss_nn_opt=loss_nn_opt,
                        final_norms=final_norms, drop_epochs=drop_epochs,
                        score_histories=all_score_histories,
                    ))

    torch.save({'results': results, 'args': vars(args),
                'checkpoints': [(ep, sd, tau) for ep, sd, tau, _ in checkpoints_with_models]},
               args.output)
    print(f"\nSaved {len(results)} results to {args.output}")

    print(f"\n{'src':>5}  {'steps':>6}  {'ss':>5}  {'λ':>6}  {'L_nn_opt':>9}  "
          f"{'survived':>9}  {'avg_drop_ep':>12}  drop_epochs")
    print('-' * 85)
    for r in results:
        survived = sum(1 for e in r['drop_epochs'] if e == -1)
        detected = [e for e in r['drop_epochs'] if e != -1]
        avg_drop = f"{np.mean(detected):.1f}" if detected else '  n/a'
        print(f"{r['src_cls']:>5}  {r['n_steps']:>6}  {r['step_size']:>5.3f}  "
              f"{r['lam']:>6.1f}  {r['loss_nn_opt']:>9.4f}  "
              f"{survived:>4}/{args.n_runs}  {avg_drop:>12}  {r['drop_epochs']}")


if __name__ == '__main__':
    main()
