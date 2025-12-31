import argparse
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.stateless import functional_call

from utils.data import load_data


class ResNet9(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 10):
        super().__init__()

        def conv_bn_relu(cin: int, cout: int, k: int = 3, s: int = 1, p: int = 1):
            return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=k, stride=s, padding=p, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=False),
            )

        self.prep = conv_bn_relu(in_channels, 64)

        self.conv1 = conv_bn_relu(64, 128, s=2)
        self.res1 = nn.Sequential(conv_bn_relu(128, 128), conv_bn_relu(128, 128))

        self.conv2 = conv_bn_relu(128, 256, s=2)
        self.conv3 = conv_bn_relu(256, 512, s=2)
        self.res2 = nn.Sequential(conv_bn_relu(512, 512), conv_bn_relu(512, 512))

        self.pool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.prep(x)
        x = self.conv1(x)
        x = x + self.res1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x + self.res2(x)
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        return self.fc(x)


def _setup_seeds(seed: int):
    seed = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _make_inclusion_split(gen: torch.Generator, m: int) -> torch.Tensor:
    m = int(m)
    if m < 2:
        raise ValueError(f"m must be >= 2 for inclusion splits, got {m}")
    perm = torch.randperm(m, generator=gen)
    half = m // 2
    include = torch.zeros(m, dtype=torch.bool)
    include[perm[:half]] = True
    return include


def _init_canaries_from_data(
    *,
    X_base: torch.Tensor,
    y_base: torch.Tensor,
    m: int,
    gen: torch.Generator,
    mode: str,
    out_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    m = int(m)
    if m < 1:
        raise ValueError(f"m must be >= 1, got {m}")

    n = int(X_base.shape[0])
    if n < m:
        raise ValueError(f"Need at least m base samples; got n={n} m={m}")

    idx = torch.randperm(n, generator=gen)[:m]
    X0 = X_base[idx].clone().detach()

    if mode == 'mislabeled':
        y0 = y_base[idx].clone().detach().long()
        y_rand = torch.randint(low=0, high=int(out_dim), size=(m,), generator=gen)
        y0 = torch.where(y_rand == y0, (y0 + 1) % int(out_dim), y_rand)
        return X0, y0

    if mode == 'random_labels':
        y0 = torch.randint(low=0, high=int(out_dim), size=(m,), generator=gen).long()
        return X0, y0

    if mode == 'random_noise':
        y0 = torch.randint(low=0, high=int(out_dim), size=(m,), generator=gen).long()
        X0 = torch.randn_like(X0, generator=gen)
        return X0, y0

    raise ValueError(f"Unknown init_canaries mode: {mode}")


def _sample_base_minibatch(
    *,
    X: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    gen: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    n = int(X.shape[0])
    b = min(int(batch_size), n)
    idx = torch.randint(low=0, high=n, size=(b,), generator=gen)
    return X[idx], y[idx]


def _unrolled_sgd_train(
    *,
    model: nn.Module,
    params: dict[str, torch.Tensor],
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    steps: int,
    lr: float,
    batch_size: int,
    gen: torch.Generator,
) -> dict[str, torch.Tensor]:
    steps = int(steps)
    if steps < 1:
        return params

    lr = float(lr)
    for i in range(steps):
        xb, yb = _sample_base_minibatch(X=X_train, y=y_train, batch_size=int(batch_size), gen=gen)
        logits = functional_call(model, params, (xb,))
        loss = F.cross_entropy(logits, yb)
        # Only retain graph on intermediate steps, not the last one
        retain = (i < steps - 1)
        grads = torch.autograd.grad(loss, list(params.values()), create_graph=True, retain_graph=retain)
        # Clip gradients for numerical stability
        grads = [torch.clamp(g, -10.0, 10.0) for g in grads]
        params = {k: v - lr * g for (k, v), g in zip(params.items(), grads)}

    return params


def _score_nll(*, model: nn.Module, params: dict[str, torch.Tensor], X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    logits = functional_call(model, params, (X,))
    return F.cross_entropy(logits, y, reduction='none')


def main():
    parser = argparse.ArgumentParser(description='Metagradient descent canary painting (surrogate ResNet-9)')

    parser.add_argument('--data_name', type=str, default='cifar10')
    parser.add_argument('--n_df', type=int, default=5000)

    parser.add_argument('--m', type=int, default=128, help='number of canaries')
    parser.add_argument('--metasteps', type=int, default=50)

    parser.add_argument('--surrogate_steps', type=int, default=50, help='number of unrolled SGD steps')
    parser.add_argument('--surrogate_lr', type=float, default=0.05)
    parser.add_argument('--surrogate_batch_size', type=int, default=256)

    parser.add_argument('--canary_lr', type=float, default=1e-1, help='learning rate for canary pixels')
    parser.add_argument('--canary_opt', type=str, default='adam', choices=['adam', 'sgd'])

    parser.add_argument('--init_canaries', type=str, default='mislabeled', choices=['mislabeled', 'random_labels', 'random_noise'])

    parser.add_argument('--objective', type=str, default='gap', choices=['gap'],
                        help='gap = mean NLL(included) - mean NLL(excluded)')

    parser.add_argument('--clamp_min', type=float, default=-4.0)
    parser.add_argument('--clamp_max', type=float, default=4.0)
    
    parser.add_argument('--print_every', type=int, default=1, help='print progress every N meta-steps')
    parser.add_argument('--early_stop_patience', type=int, default=0, help='stop if no improvement for N steps (0=disabled)')
    parser.add_argument('--early_stop_threshold', type=float, default=1e-6, help='minimum improvement threshold for early stopping')

    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--out_pt', type=str, default='debug/metagrad_canaries.pt')

    args = parser.parse_args()

    # Input validation
    if args.data_name not in ('cifar10', 'cifar100', 'mnist'):
        raise ValueError("This script currently supports cifar10/cifar100/mnist")
    
    assert args.m >= 2, f"m must be >= 2 for inclusion splits, got {args.m}"
    assert args.metasteps > 0, f"metasteps must be positive, got {args.metasteps}"
    assert args.surrogate_steps > 0, f"surrogate_steps must be positive, got {args.surrogate_steps}"
    assert args.canary_lr > 0, f"canary_lr must be positive, got {args.canary_lr}"
    assert args.surrogate_lr > 0, f"surrogate_lr must be positive, got {args.surrogate_lr}"
    assert args.surrogate_batch_size > 0, f"surrogate_batch_size must be positive, got {args.surrogate_batch_size}"
    assert args.clamp_min < args.clamp_max, f"clamp_min must be < clamp_max, got {args.clamp_min} >= {args.clamp_max}"

    _setup_seeds(int(args.seed))

    device = torch.device(str(args.device))

    X_base, y_base, out_dim = load_data(args.data_name, args.n_df, split='train')
    y_base = y_base.long()

    if X_base.ndim == 3:
        X_base = X_base.unsqueeze(1)

    gen = torch.Generator(device='cpu')
    gen.manual_seed(int(args.seed) + 1)
    meta_gen = torch.Generator(device='cpu')
    meta_gen.manual_seed(int(args.seed) + 2)

    X0, y_canary = _init_canaries_from_data(
        X_base=X_base,
        y_base=y_base,
        m=int(args.m),
        gen=gen,
        mode=str(args.init_canaries),
        out_dim=int(out_dim),
    )

    X_canary = torch.nn.Parameter(X0.to(device))
    y_canary = y_canary.to(device)

    if str(args.canary_opt) == 'adam':
        canary_opt = torch.optim.Adam([X_canary], lr=float(args.canary_lr))
    else:
        canary_opt = torch.optim.SGD([X_canary], lr=float(args.canary_lr))

    X_base = X_base.to(device)
    y_base = y_base.to(device)

    start_all = time.time()
    
    # Early stopping tracking
    best_phi = float('-inf')
    no_improve_count = 0

    for t in range(int(args.metasteps)):
        step_start = time.time()

        include_mask = _make_inclusion_split(meta_gen, int(args.m)).to(device)
        X_in = X_canary[include_mask]
        y_in = y_canary[include_mask]
        X_out = X_canary[~include_mask]
        y_out = y_canary[~include_mask]

        model = ResNet9(in_channels=int(X_base.shape[1]), num_classes=int(out_dim)).to(device)
        model.train()

        params = {k: v for k, v in model.named_parameters()}

        X_train = torch.cat([X_base, X_in], dim=0)
        y_train = torch.cat([y_base, y_in], dim=0)

        train_params = _unrolled_sgd_train(
            model=model,
            params=params,
            X_train=X_train,
            y_train=y_train,
            steps=int(args.surrogate_steps),
            lr=float(args.surrogate_lr),
            batch_size=int(args.surrogate_batch_size),
            gen=gen,
        )

        model.eval()
        nll_in = _score_nll(model=model, params=train_params, X=X_in, y=y_in)
        nll_out = _score_nll(model=model, params=train_params, X=X_out, y=y_out)

        if str(args.objective) == 'gap':
            phi = nll_in.mean() - nll_out.mean()
        else:
            raise ValueError(f"Unknown objective: {args.objective}")
        
        # Early stopping check
        phi_val = float(phi.detach().cpu())
        if args.early_stop_patience > 0:
            if phi_val > best_phi + args.early_stop_threshold:
                best_phi = phi_val
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count >= args.early_stop_patience:
                    print(f"\nEarly stopping at step {t}: no improvement for {args.early_stop_patience} steps")
                    print(f"Best phi: {best_phi:.6f}, Current phi: {phi_val:.6f}")
                    break

        canary_opt.zero_grad(set_to_none=True)
        phi.backward()
        canary_opt.step()

        with torch.no_grad():
            X_canary.clamp_(min=float(args.clamp_min), max=float(args.clamp_max))

        if (t % args.print_every) == 0:
            print(
                f"[metastep {t}/{int(args.metasteps)}] phi={float(phi.detach().cpu()):.6f} "
                f"nll_in={float(nll_in.mean().detach().cpu()):.4f} nll_out={float(nll_out.mean().detach().cpu()):.4f} "
                f"elapsed_s={time.time()-step_start:.2f}",
                flush=True,
            )
        
        # Clean up to prevent memory leak
        del model
        del params
        del train_params
        del nll_in
        del nll_out
        del phi
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.out_pt) or '.', exist_ok=True)

    payload = {
        'canaries': X_canary.detach().cpu(),
        'audit_labels': y_canary.detach().cpu(),
        'meta': {
            'data_name': str(args.data_name),
            'n_df': int(args.n_df),
            'm': int(args.m),
            'metasteps': int(args.metasteps),
            'surrogate_steps': int(args.surrogate_steps),
            'surrogate_lr': float(args.surrogate_lr),
            'surrogate_batch_size': int(args.surrogate_batch_size),
            'canary_lr': float(args.canary_lr),
            'canary_opt': str(args.canary_opt),
            'objective': str(args.objective),
            'clamp_min': float(args.clamp_min),
            'clamp_max': float(args.clamp_max),
            'seed': int(args.seed),
            'device': str(args.device),
            'wall_time_s': float(time.time() - start_all),
        },
    }

    torch.save(payload, args.out_pt)
    print(f"Saved canaries to {args.out_pt}")


if __name__ == '__main__':
    main()
