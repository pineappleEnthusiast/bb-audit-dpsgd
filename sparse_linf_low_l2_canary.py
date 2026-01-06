import argparse
import os

import numpy as np
import torch

from utils.data import load_data


def _setup_seeds(seed: int):
    seed = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _sample_reference(X: torch.Tensor, y: torch.Tensor, index: int | None, gen: torch.Generator):
    if index is not None:
        idx = int(index)
        if idx < 0 or idx >= int(X.shape[0]):
            raise ValueError(f"ref_index out of range: {idx} for n={int(X.shape[0])}")
        return X[idx:idx + 1].clone(), y[idx:idx + 1].clone()

    idx = int(torch.randint(low=0, high=int(X.shape[0]), size=(1,), generator=gen).item())
    return X[idx:idx + 1].clone(), y[idx:idx + 1].clone()


def _make_sparse_linf_perturbation(
    *,
    x: torch.Tensor,
    eps: float,
    k: int,
    gen: torch.Generator,
    per_channel: bool,
    signed: bool,
) -> torch.Tensor:
    """Return delta with L_inf == eps and sparse support.

    If k pixels are perturbed with magnitude eps, then L2 = eps*sqrt(k).

    x: shape (1,C,H,W) or (1,D)
    """
    eps = float(eps)
    k = int(k)
    if eps <= 0:
        raise ValueError(f"eps must be > 0, got {eps}")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    delta = torch.zeros_like(x)

    if x.ndim == 4:
        _, c, h, w = x.shape
        if per_channel:
            # k per channel => total perturbed = k*c
            k_eff = min(k, h * w)
            for ch in range(int(c)):
                flat_idx = torch.randperm(h * w, generator=gen)[:k_eff]
                if signed:
                    s = torch.where(torch.rand((k_eff,), generator=gen) < 0.5, -1.0, 1.0)
                else:
                    s = torch.ones((k_eff,), dtype=torch.float32)
                delta.view(1, c, h * w)[0, ch, flat_idx] = eps * s.to(delta.device)
        else:
            # k over all entries
            total = int(c * h * w)
            k_eff = min(k, total)
            flat_idx = torch.randperm(total, generator=gen)[:k_eff]
            if signed:
                s = torch.where(torch.rand((k_eff,), generator=gen) < 0.5, -1.0, 1.0)
            else:
                s = torch.ones((k_eff,), dtype=torch.float32)
            delta.view(1, total)[0, flat_idx] = eps * s.to(delta.device)

        return delta

    if x.ndim == 2:
        d = int(x.shape[1])
        k_eff = min(k, d)
        flat_idx = torch.randperm(d, generator=gen)[:k_eff]
        if signed:
            s = torch.where(torch.rand((k_eff,), generator=gen) < 0.5, -1.0, 1.0)
        else:
            s = torch.ones((k_eff,), dtype=torch.float32)
        delta[0, flat_idx] = eps * s.to(delta.device)
        return delta

    raise ValueError(f"Unsupported x.ndim={x.ndim}; expected 2 (tabular) or 4 (image)")


def _norms(delta: torch.Tensor) -> tuple[float, float]:
    flat = delta.reshape(delta.shape[0], -1)
    linf = flat.abs().max(dim=1).values.item()
    l2 = flat.norm(p=2, dim=1).item()
    return float(linf), float(l2)


def main():
    parser = argparse.ArgumentParser(description='Construct sparse canaries with high L_inf but low L2 (by top-k perturbations)')

    parser.add_argument('--data_name', type=str, default='cifar10')
    parser.add_argument('--n_df', type=int, default=5000)

    parser.add_argument('--m', type=int, default=1, help='number of canaries to generate')
    parser.add_argument('--ref_index', type=int, default=None, help='optional fixed reference sample index')

    parser.add_argument('--eps_linf', type=float, default=1.0, help='target L_inf magnitude for perturbation')
    parser.add_argument('--k', type=int, default=10, help='number of perturbed coordinates (controls L2 via eps*sqrt(k))')
    parser.add_argument('--per_channel', action='store_true', help='apply k perturbations per channel (instead of total)')
    parser.add_argument('--unsigned', action='store_true', help='use +eps only (default: random +/- eps)')

    parser.add_argument('--label_mode', type=str, default='keep', choices=['keep', 'random', 'shift'],
                        help='how to assign audit labels for canaries')

    parser.add_argument('--clamp_min', type=float, default=None)
    parser.add_argument('--clamp_max', type=float, default=None)

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out_pt', type=str, default='debug/sparse_linf_low_l2_canary.pt')

    args = parser.parse_args()

    _setup_seeds(int(args.seed))
    gen = torch.Generator(device='cpu')
    gen.manual_seed(int(args.seed) + 1)

    X, y, out_dim = load_data(args.data_name, args.n_df, split='train')
    y = y.long()

    if X.ndim == 3:
        X = X.unsqueeze(1)

    m = int(args.m)
    if m < 1:
        raise ValueError(f"--m must be >= 1, got {m}")

    Xs = []
    ys = []

    for i in range(m):
        x_ref, y_ref = _sample_reference(X, y, args.ref_index, gen)

        delta = _make_sparse_linf_perturbation(
            x=x_ref,
            eps=float(args.eps_linf),
            k=int(args.k),
            gen=gen,
            per_channel=bool(args.per_channel),
            signed=not bool(args.unsigned),
        )

        x_canary = (x_ref + delta)
        if args.clamp_min is not None or args.clamp_max is not None:
            x_canary = x_canary.clamp(min=args.clamp_min, max=args.clamp_max)

        linf, l2 = _norms(x_canary - x_ref)
        expected_l2 = float(args.eps_linf) * math.sqrt(float(args.k) * (float(x_ref.shape[1]) if bool(args.per_channel) and x_ref.ndim == 4 else 1.0))

        if str(args.label_mode) == 'keep':
            y_canary = y_ref
        elif str(args.label_mode) == 'random':
            y_canary = torch.randint(low=0, high=int(out_dim), size=(1,), generator=gen).long()
        elif str(args.label_mode) == 'shift':
            y_canary = (y_ref + 1) % int(out_dim)
        else:
            raise ValueError(f"Unknown label_mode: {args.label_mode}")

        Xs.append(x_canary)
        ys.append(y_canary)

        print(
            f"[canary {i}] linf={linf:.6f} l2={l2:.6f} expected_l2~={expected_l2:.6f} k={int(args.k)} per_channel={bool(args.per_channel)}",
            flush=True,
        )

    X_canary = torch.cat(Xs, dim=0)
    y_canary = torch.cat(ys, dim=0)

    os.makedirs(os.path.dirname(args.out_pt) or '.', exist_ok=True)
    payload = {
        'canaries': X_canary.detach().cpu(),
        'audit_labels': y_canary.detach().cpu(),
        'meta': {
            'data_name': str(args.data_name),
            'n_df': int(args.n_df),
            'm': int(m),
            'eps_linf': float(args.eps_linf),
            'k': int(args.k),
            'per_channel': bool(args.per_channel),
            'unsigned': bool(args.unsigned),
            'label_mode': str(args.label_mode),
            'clamp_min': args.clamp_min,
            'clamp_max': args.clamp_max,
            'seed': int(args.seed),
        },
    }
    torch.save(payload, args.out_pt)
    print(f"Saved canaries to {args.out_pt}")


if __name__ == '__main__':
    import math

    main()
