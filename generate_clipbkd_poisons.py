#!/usr/bin/env python3
"""
Generate clipbkd poisons.

This script creates poisons (adversarial examples) along the least variance direction of the data, scaled by the average norm with some jitter. This generates inputs that are likely to have large gradients and be sensitive to clipping defenses.

How it works:
- Compute the least variance direction using PCA on flattened training data.
- Generate k poisons by scaling this direction with average data norm plus jitter.
- Assign all poisons the label 0.

Note: Run audits using parallel_audit_model.py for parallelized evaluation.

Sample usage:
    python generate_clipbkd_poisons.py --data_name cifar10 --k 10 --out poisons.pt
"""

import argparse
import os
import time

import torch
from sklearn.decomposition import PCA

from utils.data import load_data


def _compute_least_variance_direction(X: torch.Tensor) -> torch.Tensor:
    flat_X = torch.flatten(X, start_dim=1)
    trn_x = flat_X.cpu().numpy()

    n_comps = min(trn_x.shape[0], trn_x.shape[1])
    pca = PCA(n_components=n_comps)
    pca.fit(trn_x)

    v_d = torch.from_numpy(pca.components_[-1:])
    return v_d


def generate_clipbkd_poisons(
    X: torch.Tensor,
    k: int,
    out_dim: int,
    seed: int,
    scale_jitter: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if k <= 0:
        raise ValueError(f"k must be > 0, got {k}")

    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    v_d = _compute_least_variance_direction(X)

    flat_X = torch.flatten(X, start_dim=1)
    avg_X_norm = torch.mean(torch.norm(flat_X, dim=1)).item()

    scale_noise = torch.randn((k, 1), generator=g) * scale_jitter
    scales = avg_X_norm * (1.0 + scale_noise)

    base = v_d.to(device=device, dtype=X.dtype)
    flat_poisons = scales.to(device=device, dtype=X.dtype) * base

    poison_shape = (k,) + tuple(X.shape[1:])
    X_poison = flat_poisons.reshape(poison_shape)

    if out_dim <= 0:
        raise ValueError(f"out_dim must be > 0, got {out_dim}")
    y_poison = torch.zeros((k,), dtype=torch.long, device=device)

    return X_poison, y_poison


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, required=True)
    parser.add_argument("--n_df", type=int, default=None)
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--k", type=int, required=True)

    parser.add_argument("--scale_jitter", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    X, _, out_dim = load_data(args.data_name, args.n_df, split=args.split)
    X = X.to(device)

    X_poison, y_poison = generate_clipbkd_poisons(
        X=X,
        k=args.k,
        out_dim=out_dim,
        seed=args.seed,
        scale_jitter=args.scale_jitter,
        device=device,
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    payload = {
        "canary": X_poison.detach().cpu(),
        "canary_label": y_poison.detach().cpu(),
        "X_poison": X_poison.detach().cpu(),
        "y_poison": y_poison.detach().cpu(),
        "meta": {
            "data_name": args.data_name,
            "split": args.split,
            "n_df": args.n_df,
            "k": args.k,
            "scale_jitter": args.scale_jitter,
            "seed": args.seed,
            "label": 0,
            "created_unix": time.time(),
        },
    }
    torch.save(payload, args.out)


if __name__ == "__main__":
    main()
