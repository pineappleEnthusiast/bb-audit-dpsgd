"""
Gradient Bandwidth Attack

Tracks s_t = 6th largest L∞ gradient norm within class 0 at each training epoch t,
then constructs n_canaries copies of a 1-hot gradient-space canary with L∞ norm =
min_t(s_t). This places every canary just below the defense's filtering threshold
throughout training.

Expected result: eps_lb = 0 both with and without defense, because min_t(s_t) is
tiny for natural images (dense gradients spread across 245k+ coordinates give
L∞ ≈ 0.002), so the canaries contribute negligible signal above DP noise.
"""

import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

from opacus.accountants.utils import get_noise_multiplier

from models import Models
from utils.data import load_data
from utils.dpsgd import clip_and_accum_grads, DefenseConfig


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


def train_and_find_min_6th_norm(model_name, X, y, epsilon, delta, max_grad_norm,
                                n_epochs, lr, batch_size, out_dim, defense_k,
                                target_class, device):
    """
    Train DP-SGD with top-k L∞ defense. After each epoch, record s_t = 6th largest
    L∞ gradient norm among active target_class samples. Returns min_t(s_t) and the
    hot_index (parameter with largest absolute update over full training).
    """
    device_ = torch.device(device)
    model = Models[model_name](X.shape, out_dim=out_dim).to(device_)
    xavier_init_model(model)
    init_params = {n: p.detach().clone() for n, p in model.named_parameters()}

    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    sample_rate = batch_size / len(X)
    noise_multiplier = get_noise_multiplier(
        target_epsilon=epsilon, target_delta=delta,
        sample_rate=sample_rate, epochs=n_epochs, accountant='rdp',
    )
    print(f"noise_multiplier={noise_multiplier:.4f}")

    defense_cfg_proto = DefenseConfig(score_fn='grad_norm_unclipped', score_norm='linf')

    dataset = _IndexedDataset(X, y)
    n = len(dataset)
    scores    = np.zeros(n, dtype=np.float32)
    drop_mask = np.zeros(n, dtype=np.int8)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(0),
                        num_workers=0, drop_last=False)

    sixth_largest_per_epoch = []

    for epoch in range(n_epochs):
        t0 = time.time()
        optimizer.zero_grad()

        for curr_X, curr_y, global_indices in loader:
            curr_X = curr_X.to(device_)
            curr_y = curr_y.to(device_)
            global_indices = global_indices.to(device_)

            local_dm = drop_mask[global_indices.cpu().numpy()]
            defense_cfg = deepcopy(defense_cfg_proto)
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
                    if param.grad is None:
                        param.grad = g
                    else:
                        param.grad.copy_(g)
            optimizer.step()
            optimizer.zero_grad()

        # s_t: 6th largest L∞ norm among active target_class samples this epoch
        active_target = np.where(
            (y.cpu().numpy() == target_class) & (drop_mask == 0)
        )[0]
        if len(active_target) >= defense_k + 1:
            cls_scores = scores[active_target]
            sorted_scores = np.sort(cls_scores)[::-1]
            s_t = float(sorted_scores[defense_k])  # index defense_k = (k+1)-th largest
            sixth_largest_per_epoch.append(s_t)
            print(f"  epoch {epoch:3d}: s_t={s_t:.6f}  active={len(active_target)}  "
                  f"({time.time()-t0:.1f}s)")
        else:
            print(f"  epoch {epoch:3d}: too few active class-{target_class} samples, skipping")

        # End-of-epoch defense: drop top-k per class
        active_mask = torch.from_numpy(drop_mask == 0)
        for cls in torch.unique(y).tolist():
            cls_idx = ((y == int(cls)) & active_mask).nonzero(as_tuple=True)[0]
            if len(cls_idx) == 0:
                continue
            cls_sc = torch.from_numpy(scores[cls_idx.numpy()])
            _, topk = torch.topk(cls_sc, min(defense_k, len(cls_sc)))
            drop_mask[cls_idx[topk].numpy()] = 1
        scores.fill(0)

    # Hot index: parameter with largest absolute movement over full training
    final_params = model.state_dict()
    flat_update = torch.cat([
        (final_params[n] - init_params[n]).view(-1) for n in init_params
    ])
    hot_index = int(flat_update.abs().argmax().item())
    print(f"\nhot_index={hot_index}  |Δθ|={flat_update[hot_index].abs().item():.6f}")

    min_norm = float(min(sixth_largest_per_epoch)) if sixth_largest_per_epoch else None
    if min_norm is not None:
        print(f"min_t(s_t) = {min_norm:.6f}  (over {len(sixth_largest_per_epoch)} epochs)")
    return min_norm, hot_index


def make_1hot_gradient(model, hot_index, norm_value, device):
    """1-hot gradient dict: value=norm_value at hot_index, zero elsewhere."""
    offset = 0
    crafted = {}
    for name, param in model.named_parameters():
        g = torch.zeros_like(param)
        n_elem = param.numel()
        if offset <= hot_index < offset + n_elem:
            g.view(-1)[hot_index - offset] = norm_value
        crafted[name] = g.unsqueeze(0)
        offset += n_elem
    return crafted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name',   type=str,   default='mnist')
    parser.add_argument('--model_name',  type=str,   default='cnn')
    parser.add_argument('--out_dim',     type=int,   default=None)
    parser.add_argument('--n_epochs',    type=int,   default=100)
    parser.add_argument('--lr',          type=float, default=3.0)
    parser.add_argument('--batch_size',  type=int,   default=4000)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--epsilon',     type=float, default=10.0)
    parser.add_argument('--delta',       type=float, default=1e-5)
    parser.add_argument('--defense_k',   type=int,   default=5)
    parser.add_argument('--target_class',type=int,   default=0,
                        help='Class whose gradient norms are tracked for s_t')
    parser.add_argument('--n_canaries',  type=int,   default=500,
                        help='Number of 1-hot canary copies to generate')
    parser.add_argument('--seed',        type=int,   default=0)
    parser.add_argument('--device',      type=str,   default='cuda:0')
    parser.add_argument('--output_dir',  type=str,   default='grad_bandwidth_attack')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Loading {args.data_name}...")
    X, y, out_dim = load_data(args.data_name, n_df=None)
    if args.out_dim is not None:
        out_dim = args.out_dim
    X = X.float()
    y = y.long()
    print(f"  {len(X)} samples, out_dim={out_dim}, input_shape={tuple(X.shape[1:])}")

    print(f"\nTraining to find min_t(s_t)...")
    min_norm, hot_index = train_and_find_min_6th_norm(
        model_name=args.model_name, X=X, y=y,
        epsilon=args.epsilon, delta=args.delta,
        max_grad_norm=args.max_grad_norm,
        n_epochs=args.n_epochs, lr=args.lr, batch_size=args.batch_size,
        out_dim=out_dim, defense_k=args.defense_k,
        target_class=args.target_class, device=args.device,
    )

    if min_norm is None:
        print("No valid s_t measurements — exiting.")
        return

    # Build n_canaries identical 1-hot gradients at hot_index with norm=min_norm
    print(f"\nBuilding {args.n_canaries} canaries at hot_index={hot_index}, norm={min_norm:.6f}...")
    dummy = Models[args.model_name](X.shape, out_dim=out_dim)
    total_params = sum(p.numel() for p in dummy.parameters())
    print(f"  Total parameters: {total_params}")

    single_grad = make_1hot_gradient(dummy, hot_index, min_norm, args.device)
    gradients = [single_grad] * args.n_canaries  # all identical copies

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'gradient_space_canaries.pt'
    torch.save({
        'gradients':  gradients,
        'n_canaries': args.n_canaries,
        'norm':       min_norm,
        'hot_index':  hot_index,
        'target_class': args.target_class,
    }, out_path)
    print(f"Saved {args.n_canaries} canaries to {out_path}")

    # Quick verification
    flat = torch.cat([g.view(-1) for g in single_grad.values()])
    print(f"  Canary L∞={flat.abs().max().item():.6f}  L2={flat.norm().item():.6f}")
    print(f"\nTo audit (no defense):")
    print(f"  torchrun ... parallel_audit_multi_canary.py "
          f"--gradient_space_canary_pt {out_path} "
          f"--target_type gradient_space_canary "
          f"--gradient_space_score_fn hot_param")


if __name__ == '__main__':
    main()
