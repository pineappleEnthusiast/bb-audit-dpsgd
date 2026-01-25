import argparse
import copy
import os
import time

import dill
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset

from models import Models
from utils.data import load_data
from utils.audit import compute_eps_lower_from_mia
from utils.audit import compute_eps_lower_from_mia_given_t
from utils.dpsgd import clip_and_accum_grads, DefenseConfig
from opacus.accountants.utils import get_noise_multiplier

from parallel_audit_model import (
    AugmentationFunction,
    craft_clipbkd,
    craft_gradient,
    fgsm_attack,
    init_wideresnet,
    xavier_init_model,
)


class IndexedTensorDataset(torch.utils.data.Dataset):
    """A dataset that includes the index of each sample."""

    def __init__(self, *tensors):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors) + (index,)

    def __len__(self):
        return self.tensors[0].size(0)


def _setup_seeds(seed: int):
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def _resolve_device(device_str: str | None):
    if device_str is None:
        return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_str)


def _make_blank_canaries(X_ref: torch.Tensor, y_ref: torch.Tensor, n_canaries: int, blank_alpha: float):
    if X_ref.ndim < 2:
        raise ValueError(f"Unexpected X_ref.ndim={X_ref.ndim}")

    ref = X_ref[[0]].clone()
    blank = torch.zeros_like(ref)
    x = (1.0 - float(blank_alpha)) * blank + float(blank_alpha) * ref

    if y_ref.ndim == 0:
        y0 = y_ref.view(1).clone()
    else:
        y0 = y_ref[[0]].clone()

    Xc = x.repeat(int(n_canaries), *([1] * (x.ndim - 1)))
    yc = y0.repeat(int(n_canaries))
    return Xc, yc


def _load_canaries_from_pt_dict(pt_path: str, ref_X: torch.Tensor):
    """Load canaries from a .pt file.

    Supported schemas:
    - {'canaries': Tensor[N,...], 'audit_labels': Tensor[N] | list[int]}
    - {'canaries': [{'canary': Tensor[...], 'audit_label': int}, ...]}
    - {'canary': Tensor[...], 'audit_label': int}  (single canary)

    Also supports optional 'init_model' in the dict (state_dict) for reproducible init.

    Returns: (X_canary, y_canary, meta)
    """
    if pt_path is None:
        raise ValueError("pt_path must not be None")
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"Canary pt file not found: {pt_path}")

    d = torch.load(pt_path, map_location='cpu')
    if not isinstance(d, dict):
        raise ValueError(f"Expected {pt_path} to be a dict")

    init_model_state = d.get('init_model', None)

    def _label_key_candidates(*, plural: bool) -> list[str]:
        if plural:
            return [
                'audit_labels',
                'target_labels',
                'canary_labels',
                'labels',
                'true_labels',
            ]
        return [
            'target_label',
            'canary_label',
            'label',
            'true_label',
            'audit_label',
        ]

    def _extract_labels_from_dict(payload: dict, n: int) -> torch.Tensor:
        y_val = None
        for k in _label_key_candidates(plural=True):
            if k in payload:
                y_val = payload[k]
                break
        if y_val is None:
            # Match parallel_audit_model default label when not provided.
            return torch.full((n,), 9, dtype=torch.long)

        if torch.is_tensor(y_val):
            y_list = y_val.detach().cpu().view(-1).tolist()
        elif isinstance(y_val, (list, tuple)):
            y_list = list(y_val)
        else:
            # scalar broadcast
            y_list = [y_val] * n

        y = torch.tensor([int(x) for x in y_list], dtype=torch.long)
        if y.numel() == 1 and n > 1:
            y = y.repeat(n)
        if y.numel() != n:
            raise ValueError(f"Expected {n} labels, got {y.numel()}")
        return y

    def _extract_single_label_from_dict(payload: dict) -> int:
        for k in _label_key_candidates(plural=False):
            if k in payload:
                v = payload[k]
                if torch.is_tensor(v):
                    return int(v.detach().cpu().view(-1)[0].item())
                return int(v)
        return 9

    if 'canaries' in d and torch.is_tensor(d['canaries']):
        X_canary = d['canaries']
        if not torch.is_tensor(X_canary):
            raise ValueError("Expected 'canaries' to be a Tensor")
        y_canary = _extract_labels_from_dict(d, int(X_canary.shape[0]))
    elif 'canaries' in d and isinstance(d['canaries'], (list, tuple)):
        items = d['canaries']
        Xs = []
        ys = []
        for item in items:
            if not isinstance(item, dict) or 'canary' not in item:
                raise ValueError("Expected each entry in 'canaries' to be a dict with key 'canary'")
            x = item['canary']
            if not torch.is_tensor(x):
                raise ValueError("Expected item['canary'] to be a Tensor")
            y = _extract_single_label_from_dict(item)
            if x.ndim == ref_X.ndim - 1:
                x = x.unsqueeze(0)
            Xs.append(x)
            ys.append(y)
        X_canary = torch.cat(Xs, dim=0)
        y_canary = torch.tensor(ys, dtype=torch.long)
    elif 'canary' in d:
        x = d['canary']
        if not torch.is_tensor(x):
            raise ValueError("Expected 'canary' to be a Tensor")
        if x.ndim == ref_X.ndim - 1:
            x = x.unsqueeze(0)
        X_canary = x
        # Single-canary schema uses scalar label keys; if X_canary contains a batch, we also
        # accept plural label keys and broadcast scalar labels.
        if int(X_canary.shape[0]) > 1:
            y_canary = _extract_labels_from_dict(d, int(X_canary.shape[0]))
        else:
            y_canary = torch.tensor([_extract_single_label_from_dict(d)], dtype=torch.long)
    else:
        raise ValueError(
            "Unrecognized canary .pt schema. Supported schemas:\n"
            "  1) {'canaries': Tensor[N,...], 'audit_labels': Tensor[N]|list[int]}\n"
            "  2) {'canaries': [{'canary': Tensor, 'audit_label': int}, ...]}\n"
            "  3) {'canary': Tensor, 'audit_label': int}"
        )

    if X_canary.ndim == ref_X.ndim - 1:
        X_canary = X_canary.unsqueeze(0)
    if X_canary.ndim != ref_X.ndim:
        raise ValueError(f"Loaded canaries have ndim={X_canary.ndim}, expected {ref_X.ndim}")
    if tuple(X_canary.shape[1:]) != tuple(ref_X.shape[1:]):
        raise ValueError(f"Loaded canaries have sample shape {tuple(X_canary.shape[1:])}, expected {tuple(ref_X.shape[1:])}")
    if y_canary.ndim != 1 or y_canary.shape[0] != X_canary.shape[0]:
        raise ValueError(f"audit_labels must have shape (N,), got {tuple(y_canary.shape)} for N={X_canary.shape[0]}")

    meta = {
        'canary_pt_path': pt_path,
        'n_canaries_loaded': int(X_canary.shape[0]),
        'has_init_model_state': init_model_state is not None,
        'init_model_state': init_model_state,
    }
    return X_canary, y_canary, meta


def _make_sanity_check_canaries(X_out: torch.Tensor, y_out: torch.Tensor, n_canaries: int):
    # Use the last sample from D- replicated.
    x = X_out[-1].unsqueeze(0).clone() if X_out.ndim >= 2 else X_out[-1].view(1)
    y = y_out[-1].view(1).clone()
    Xc = x.repeat(int(n_canaries), *([1] * (x.ndim - 1)))
    yc = y.repeat(int(n_canaries))
    return Xc, yc


def _make_canaries_mislabeled(
    X_ref: torch.Tensor,
    y_ref: torch.Tensor,
    *,
    n_canaries: int,
    out_dim: int,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create `n_canaries` canaries by taking the last `n_canaries` samples and relabeling them.

    Each canary is randomly mislabeled to a class different from its true label.
    When possible, avoids reusing labels to maximize diversity.
    """
    out_dim = int(out_dim)

    if int(n_canaries) < 1:
        raise ValueError(f"n_canaries must be >= 1, got {n_canaries}")

    if out_dim <= 1:
        raise ValueError(f"out_dim must be > 1 to mislabel, got {out_dim}")

    if X_ref.shape[0] < int(n_canaries):
        raise ValueError(f"Need at least n_canaries samples. Got n={int(X_ref.shape[0])} n_canaries={int(n_canaries)}")

    Xc = X_ref[-int(n_canaries):].clone()
    y_true = y_ref[-int(n_canaries):].clone().long().view(-1)

    # Create random generator for reproducible mislabeling
    rng = np.random.default_rng(seed)

    y_mis = []

    for true_label in y_true.tolist():
        # Get available labels (all except true label)
        available = [i for i in range(out_dim) if i != true_label]

        # Randomly choose from available labels
        chosen = rng.choice(available)
        y_mis.append(chosen)

    y_mis = torch.tensor(y_mis, dtype=torch.long)
    return Xc, y_mis


def _score_canaries(model: torch.nn.Module, X_canary: torch.Tensor, y_canary: torch.Tensor) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        dev = next(model.parameters()).device
        logits = model(X_canary.to(dev))
        # Use negative cross-entropy: higher score = lower loss = better prediction
        per = -F.cross_entropy(logits, y_canary.to(dev), reduction='none')
        return per.detach().cpu().numpy().astype(np.float32)


def _audit_from_scores(
    scores_in: np.ndarray,
    scores_out: np.ndarray,
    alpha: float,
    delta: float,
    holdout_audit: bool,
    seed: int = 0,
):
    # scores_* are arrays with shape (n_reps_half,) representing the audit statistic per model.
    if scores_in.shape[0] != scores_out.shape[0]:
        raise ValueError(f"Expected same number of in/out reps, got {scores_in.shape[0]} and {scores_out.shape[0]}")

    n = int(scores_in.shape[0])
    
    if holdout_audit:
        if n < 2:
            raise ValueError("holdout_audit requires at least 2 reps per world")
        
        # Use random sampling for holdout split to avoid ordering effects
        np.random.seed(seed)  # Use same seed for reproducibility
        indices = np.random.permutation(n)
        threshold_indices = indices[:n // 2]
        holdout_indices = indices[n // 2:]
        
        t_scores_in = scores_in[threshold_indices]
        t_scores_out = scores_out[threshold_indices]
        hold_scores_in = scores_in[holdout_indices]
        hold_scores_out = scores_out[holdout_indices]
    else:
        # No holdout - use all data for threshold selection
        t_scores_in = scores_in
        t_scores_out = scores_out

    mia_scores = np.concatenate([t_scores_in, t_scores_out]).astype(np.float32)
    mia_labels = np.concatenate([
        np.ones(t_scores_in.shape[0], dtype=np.int64),
        np.zeros(t_scores_out.shape[0], dtype=np.int64),
    ])

    t, emp_eps = compute_eps_lower_from_mia(mia_scores, mia_labels, alpha, delta, 'GDP', n_procs=1)

    if holdout_audit:
        hold_mia_scores = np.concatenate([hold_scores_in, hold_scores_out]).astype(np.float32)
        hold_mia_labels = np.concatenate([
            np.ones(hold_scores_in.shape[0], dtype=np.int64),
            np.zeros(hold_scores_out.shape[0], dtype=np.int64),
        ])
        emp_eps = compute_eps_lower_from_mia_given_t(hold_mia_scores, hold_mia_labels, alpha, delta, t, 'GDP')

    return float(emp_eps), float(t), mia_scores, mia_labels


def test_model(model, X, y, batch_size=128):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    X = X.to(device)
    y = y.to(device)

    test_loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=False)

    model.eval()
    acc = 0
    total = 0
    with torch.no_grad():
        for curr_X, curr_y in test_loader:
            curr_X = curr_X.to(device)
            curr_y = curr_y.to(device)
            curr_y_hat = torch.argmax(model(curr_X), dim=1)
            acc += torch.sum(curr_y_hat == curr_y).cpu().item()
            total += len(curr_y)

    model.train()
    return acc / total if total > 0 else 0.0


def train_model_multi_canary(
    *,
    model_name: str,
    X: torch.Tensor,
    y: torch.Tensor,
    epsilon: float | None,
    delta: float | None,
    max_grad_norm: float | None,
    n_epochs: int,
    lr: float,
    block_size: int,
    batch_size: int,
    init_model: torch.nn.Module | None,
    out_dim: int,
    aug_mult: int,
    defense: bool,
    defense_k: int,
    defense_apply_ascent: bool,
    defense_filter_every: int,
    defense_score_fn: str,
    defense_score_norm: str,
    defense_global_filter: bool,
    device: str,
    generator: torch.Generator | None,
    dl_generator: torch.Generator | None,
    num_workers: int,
    persistent_workers: bool,
    canary_indices: np.ndarray | None,
    is_gradient_space_canary: bool = False,
    global_idx_to_grad: dict | None = None,
    loss_volatility_k: int = 5,
    grad_norm_percentile_k: int = 20,
    grad_dir_volatility_k: int = 5,
    grad_dir_proj_dim: int = 64,
    grad_dir_proj_seed: int = 0,
    rand_proj_var_m: int = 10,
    rand_proj_var_seed: int = 0,
    maxmin_proj_k: int = 10,
    maxmin_proj_seed: int = 0,
    grad_rank_mode: str = 'effdim',
    grad_rank_eps: float = 1e-12,
    grad_accel_proj_dim: int = 64,
    grad_accel_proj_seed: int = 0,
    grad_jerk_proj_dim: int = 64,
    grad_jerk_proj_seed: int = 0,
    dir_unique_k: int = 5,
    alignment_proj_k: int = 10,
    alignment_proj_seed: int = 0,
    grad_scatter_k: int = 5,
    rank: int = 0,
):
    """Train a model with the same core logic as parallel_audit_model.train_model,
    but extended to track multiple canaries.

    Returns: (model, drop_mask, defense_stats)
    where defense_stats contains:
      - canary_indices: np.int64[NC]
      - canary_drop_epochs: np.int32[NC] (epoch when canary transitioned 1->2, else -1)
      - canary_drop_ratio_events: list[(epoch:int, dropped_ratio:float, dropped_count:int)]
    """

    canary_indices_np = None
    if canary_indices is not None:
        canary_indices_np = np.asarray(canary_indices, dtype=np.int64)
        if canary_indices_np.ndim != 1:
            raise ValueError(f"canary_indices must be 1D, got shape {canary_indices_np.shape}")
        if canary_indices_np.size == 0:
            raise ValueError("canary_indices must be non-empty when provided")
    # Note: defense can run without canary tracking (e.g., for OUT world).

    # Device
    dev = torch.device(device)
    if dev.type == 'cuda':
        torch.cuda.set_device(dev)

    # Init model
    if init_model is None:
        model = Models[model_name](X.shape, out_dim=out_dim)
        if model_name == 'cnn':
            xavier_init_model(model)
        else:
            init_wideresnet(model)
    else:
        model = copy.deepcopy(init_model)
    model = model.to(dev)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=float(lr))

    # DP noise
    if epsilon is not None:
        sample_rate = float(batch_size) / float(len(X))
        noise_multiplier = get_noise_multiplier(
            target_epsilon=float(epsilon),
            target_delta=float(delta) if delta is not None else 1e-5,
            sample_rate=sample_rate,
            epochs=int(n_epochs),
            accountant='rdp',
        )
    else:
        noise_multiplier = 0.0

    # Aug
    if len(X.shape) > 2 and int(aug_mult) > 1:
        aug_fn = AugmentationFunction(X.shape[2], X.shape[1])
    else:
        aug_fn = None

    # Dataset
    dataset = IndexedTensorDataset(X, y)
    scores = np.zeros(len(dataset), dtype=np.float32)
    drop_mask = np.zeros(len(dataset), dtype=np.int8)

    sampler = torch.utils.data.RandomSampler(dataset, replacement=False, generator=generator)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(batch_size),
        sampler=sampler,
        pin_memory=True,
        num_workers=int(num_workers),
        persistent_workers=bool(persistent_workers) if int(num_workers) > 0 else False,
        drop_last=False,
        generator=dl_generator,
    )

    # Multi-canary defense tracking
    canary_drop_epochs = None
    canary_drop_ratio_events: list[tuple[int, float, int]] = []
    if canary_indices_np is not None:
        canary_drop_epochs = np.full((canary_indices_np.size,), -1, dtype=np.int32)

    # State variables for scoring functions
    prev_params = None
    prev_delta_theta = None
    theta0_params = None
    prev_losses = None
    loss_hist = None
    loss_hist_pos = None
    grad_norm_hist = None
    grad_norm_hist_pos = None
    grad_dir_hist = None
    grad_dir_hist_pos = None
    grad_dir_proj = None
    rand_proj_mat = None
    maxmin_proj_mat = None
    grad_accel_hist = None
    grad_accel_hist_pos = None
    grad_accel_proj = None
    grad_jerk_hist = None
    grad_jerk_hist_pos = None
    grad_jerk_proj = None
    dir_unique_hist = None
    dir_unique_hist_pos = None
    alignment_proj_mat = None

    for epoch in range(int(n_epochs)):
        epoch_start = time.time()
        optimizer.zero_grad()
        print(f"Epoch: {epoch} (Active samples: {int((drop_mask == 0).sum())}/{len(drop_mask)})", end='', flush=True)

        for _, (curr_X, curr_y, global_indices) in enumerate(loader):
            curr_X = curr_X.to(dev, non_blocking=True)
            curr_y = curr_y.to(dev, non_blocking=True)
            global_indices = global_indices.to(dev, non_blocking=True)

            # Initialize history-based scoring function state
            if defense_score_fn == 'loss_momentum' and prev_losses is None:
                prev_losses = np.full((len(dataset),), np.nan, dtype=np.float32)

            if defense_score_fn == 'loss_volatility' and loss_hist is None:
                k = int(loss_volatility_k)
                if k <= 0:
                    raise ValueError(f"loss_volatility_k must be > 0, got {k}")
                loss_hist = np.full((len(dataset), k), np.nan, dtype=np.float32)
                loss_hist_pos = np.zeros((len(dataset),), dtype=np.int64)

            if defense_score_fn == 'grad_norm_percentile' and grad_norm_hist is None:
                k = int(grad_norm_percentile_k)
                if k <= 0:
                    raise ValueError(f"grad_norm_percentile_k must be > 0, got {k}")
                grad_norm_hist = np.full((len(dataset), k), np.nan, dtype=np.float32)
                grad_norm_hist_pos = np.zeros((len(dataset),), dtype=np.int64)

            if defense_score_fn == 'grad_dir_volatility' and grad_dir_hist is None:
                k = int(grad_dir_volatility_k)
                if k <= 0:
                    raise ValueError(f"grad_dir_volatility_k must be > 0, got {k}")
                # Note: grad_dir_proj will be created lazily on first batch when we know the actual gradient dimensions
                grad_dir_hist = np.full((len(dataset), k, int(grad_dir_proj_dim)), np.nan, dtype=np.float32)
                grad_dir_hist_pos = np.zeros((len(dataset),), dtype=np.int64)

            if defense_score_fn == 'norm_x_dir_uniqueness' and dir_unique_hist is None:
                k = int(dir_unique_k)
                if k <= 0:
                    raise ValueError(f"dir_unique_k must be > 0, got {k}")
                # Note: grad_dir_proj will be created lazily on first batch
                dir_unique_hist = np.full((len(dataset), k, int(grad_dir_proj_dim)), np.nan, dtype=np.float32)
                dir_unique_hist_pos = np.zeros((len(dataset),), dtype=np.int64)

            if defense_score_fn == 'rand_proj_var' and rand_proj_mat is None:
                pass  # Will be created in dpsgd.py

            if defense_score_fn == 'maxmin_proj_ratio' and maxmin_proj_mat is None:
                pass  # Will be created in dpsgd.py

            if defense_score_fn == 'alignment_with_rand_proj' and alignment_proj_mat is None:
                pass  # Will be created in dpsgd.py

            if defense_score_fn == 'grad_accel' and grad_accel_hist is None:
                # Keep a 3-step history for discrete second difference.
                # Note: grad_accel_proj will be created lazily on first batch
                grad_accel_hist = np.full((len(dataset), 3, int(grad_accel_proj_dim)), np.nan, dtype=np.float32)
                grad_accel_hist_pos = np.zeros((len(dataset),), dtype=np.int64)

            if defense_score_fn == 'grad_jerk' and grad_jerk_hist is None:
                # Keep a 4-step history for discrete third difference.
                # Note: grad_jerk_proj will be created lazily on first batch
                grad_jerk_hist = np.full((len(dataset), 4, int(grad_jerk_proj_dim)), np.nan, dtype=np.float32)
                grad_jerk_hist_pos = np.zeros((len(dataset),), dtype=np.int64)

            if defense_score_fn == 'cos_update' and prev_params is None:
                prev_params = {n: p.detach().clone() for n, p in model.named_parameters()}

            if defense_score_fn == 'cos_theta0' and theta0_params is None:
                theta0_params = {n: p.detach().clone() for n, p in model.named_parameters()}

            if defense_score_fn == 'norm_x_trajectory_orth' and theta0_params is None:
                theta0_params = {n: p.detach().clone() for n, p in model.named_parameters()}

            # Compute parameter differences for trajectory-based scoring functions
            curr_params = {n: p.detach() for n, p in model.named_parameters()}
            if prev_params is not None:
                prev_delta_theta = {n: curr_params[n] - prev_params[n] for n in prev_params.keys()}
            else:
                prev_delta_theta = None

            if theta0_params is not None:
                theta_t_minus_theta0 = {n: curr_params[n] - theta0_params[n] for n in theta0_params.keys()}
            else:
                theta_t_minus_theta0 = None

            defense_cfg = DefenseConfig(
                score_fn=defense_score_fn,
                score_norm=defense_score_norm,
                delta_theta=prev_delta_theta,
                theta_t_minus_theta0=theta_t_minus_theta0,
                grad_norm_hist=grad_norm_hist,
                grad_norm_hist_pos=grad_norm_hist_pos,
                grad_norm_percentile_k=int(grad_norm_percentile_k),
                grad_dir_hist=grad_dir_hist,
                grad_dir_hist_pos=grad_dir_hist_pos,
                grad_dir_volatility_k=int(grad_dir_volatility_k),
                grad_dir_proj=grad_dir_proj,
                rand_proj_mat=rand_proj_mat,
                rand_proj_var_m=int(rand_proj_var_m),
                maxmin_proj_mat=maxmin_proj_mat,
                maxmin_proj_k=int(maxmin_proj_k),
                grad_rank_mode=str(grad_rank_mode),
                grad_rank_eps=float(grad_rank_eps),
                grad_accel_hist=grad_accel_hist,
                grad_accel_hist_pos=grad_accel_hist_pos,
                grad_accel_proj=grad_accel_proj,
                grad_jerk_hist=grad_jerk_hist,
                grad_jerk_hist_pos=grad_jerk_hist_pos,
                alignment_proj_mat=alignment_proj_mat,
                alignment_proj_k=int(alignment_proj_k),
                grad_jerk_proj=grad_jerk_proj,
                dir_unique_hist=dir_unique_hist,
                dir_unique_hist_pos=dir_unique_hist_pos,
                dir_unique_k=int(dir_unique_k),
                grad_scatter_k=int(grad_scatter_k)
            )

            curr_accumulated_gradients, scores = clip_and_accum_grads(
                model,
                curr_X,
                curr_y,
                optimizer,
                criterion,
                max_grad_norm,
                drop_mask=drop_mask[global_indices.cpu().numpy()] if drop_mask is not None else None,
                block_size=min(int(block_size), int(batch_size)),
                scores=scores,
                device=dev,
                global_indices=global_indices,
                aug_mult=int(aug_mult),
                aug_fn=aug_fn,
                world_size=1,
                rank=0,
                batch_size=int(batch_size),
                is_gradient_space_canary=is_gradient_space_canary,
                global_idx_to_grad=global_idx_to_grad,
                canary_indices=canary_indices_np,
                defense_cfg=defense_cfg,
                defense_apply_ascent=bool(defense_apply_ascent),
            )

            # Update projection matrices if they were lazily created in dpsgd.py
            if defense_cfg.grad_dir_proj is not None:
                grad_dir_proj = defense_cfg.grad_dir_proj
            if defense_cfg.rand_proj_mat is not None:
                rand_proj_mat = defense_cfg.rand_proj_mat
            if defense_cfg.maxmin_proj_mat is not None:
                maxmin_proj_mat = defense_cfg.maxmin_proj_mat
            if defense_cfg.alignment_proj_mat is not None:
                alignment_proj_mat = defense_cfg.alignment_proj_mat
            if defense_cfg.grad_accel_proj is not None:
                grad_accel_proj = defense_cfg.grad_accel_proj
            if defense_cfg.grad_jerk_proj is not None:
                grad_jerk_proj = defense_cfg.grad_jerk_proj

            # Transition ascent->dropped ONLY for samples in the current batch
            # This ensures samples marked at end of epoch get gradient ascent applied throughout the next epoch
            batch_indices = global_indices.cpu().numpy()
            batch_drop_mask = drop_mask[batch_indices]
            samples_to_transition = batch_indices[batch_drop_mask == 1]
            
            # Track canaries transitioning 1 -> 2 (in this implementation the 1->2 happens here)
            if defense and canary_indices_np is not None and canary_drop_epochs is not None and len(samples_to_transition) > 0:
                newly_dropped = np.intersect1d(samples_to_transition, canary_indices_np, assume_unique=False)
                if newly_dropped.size > 0:
                    pos = np.nonzero(np.isin(canary_indices_np, newly_dropped))[0]
                    unset = canary_drop_epochs[pos] < 0
                    if np.any(unset):
                        canary_drop_epochs[pos[unset]] = int(epoch)
                        dropped_count = int((canary_drop_epochs >= 0).sum())
                        dropped_ratio = dropped_count / float(canary_indices_np.size)
                        canary_drop_ratio_events.append((int(epoch), float(dropped_ratio), int(dropped_count)))
            
            # Only transition samples that were in this batch
            drop_mask[samples_to_transition] = 2

            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name not in curr_accumulated_gradients:
                        continue
                    grad = curr_accumulated_gradients[name].to(dev)

                    # Add DP noise to the sum of clipped gradients (before averaging)
                    if noise_multiplier > 0 and max_grad_norm is not None:
                        noise_std = float(noise_multiplier) * float(max_grad_norm)
                        grad.add_(noise_std * torch.randn_like(grad))
                    
                    # Average the noisy gradient sum
                    batch_size_in = int(curr_X.shape[0])
                    grad.div_(float(batch_size_in))

                    if param.grad is None:
                        param.grad = grad.clone()
                    else:
                        param.grad.copy_(grad)

            optimizer.step()
            optimizer.zero_grad()

            # Update prev_params for cos_update scoring function
            if defense_score_fn == 'cos_update' and prev_params is not None:
                curr_params = {n: p.detach() for n, p in model.named_parameters()}
                prev_delta_theta = {n: curr_params[n] - prev_params[n] for n in prev_params.keys()}
                prev_params = {n: curr_params[n].clone() for n in prev_params.keys()}

        epoch_time = time.time() - epoch_start
        print(f" | Time: {epoch_time:.2f}s")

        # Defense marking happens at epoch end
        if defense and (epoch % defense_filter_every == 0):
            k = int(defense_k)
            active_mask = torch.from_numpy(drop_mask == 0)

            # Track how many samples are marked before this epoch's filtering
            n_marked_before = int((drop_mask == 1).sum())

            if defense_global_filter:
                # GLOBAL FILTERING: Filter top k scores across entire dataset (not per-class)
                active_indices = active_mask.nonzero(as_tuple=True)[0]
                
                if len(active_indices) > 0:
                    active_scores = torch.tensor(scores[active_indices.cpu().numpy()], device=y.device)
                    _, topk_indices = torch.topk(active_scores, min(k, len(active_scores)))
                    
                    topk_global_indices = active_indices[topk_indices]
                    
                    dropped_indices = topk_global_indices.cpu().numpy()
                    drop_mask[dropped_indices] = 1
            else:
                # PER-CLASS FILTERING: Filter top k per class
                unique_classes = torch.unique(y).cpu()
                for cls in unique_classes:
                    cls_indices = ((y.cpu() == cls.item()) & active_mask).nonzero(as_tuple=True)[0]
                    if len(cls_indices) == 0:
                        continue
                    cls_scores = torch.tensor(scores[cls_indices.cpu().numpy()], device=y.device)
                    _, topk_indices = torch.topk(cls_scores, min(k, len(cls_scores)))
                    topk_global_indices = cls_indices[topk_indices]
                    dropped_indices = topk_global_indices.cpu().numpy()
                    drop_mask[dropped_indices] = 1

            # Check how many samples were newly marked in this epoch
            n_marked_after = int((drop_mask == 1).sum())
            n_newly_marked = n_marked_after - n_marked_before
            
            if n_newly_marked > 0:
                if canary_indices_np is not None:
                    # Check which canaries were newly marked in this epoch
                    newly_marked_indices = np.where(drop_mask == 1)[0]
                    newly_marked_canaries = np.intersect1d(newly_marked_indices, canary_indices_np)
                    n_canaries_marked_this_epoch = len(newly_marked_canaries)
                    
                    # Count total canaries that have been marked or dropped (drop_mask >= 1)
                    total_canaries_filtered = np.sum(np.isin(canary_indices_np, np.where(drop_mask >= 1)[0]))
                    canary_fraction = total_canaries_filtered / len(canary_indices_np) if len(canary_indices_np) > 0 else 0
                    
                    if rank == 0:
                        if n_canaries_marked_this_epoch > 0:
                            print(f"  [Defense] Marked {n_newly_marked} samples for filtering (including {n_canaries_marked_this_epoch} canaries, {canary_fraction:.1%} of canaries filtered so far)")
                            
                            # Show defense scores for newly marked canaries
                            newly_marked_canary_scores = scores[newly_marked_canaries]
                            print(f"  [Defense] Newly marked canary scores: min={newly_marked_canary_scores.min():.6f}, max={newly_marked_canary_scores.max():.6f}, mean={newly_marked_canary_scores.mean():.6f}")
                        else:
                            print(f"  [Defense] Marked {n_newly_marked} samples for filtering ({canary_fraction:.1%} of canaries filtered so far)")
                        
                        # Show score statistics for all canaries vs all samples
                        canary_scores = scores[canary_indices_np]
                        all_scores = scores[scores > 0]  # Only non-zero scores
                        if len(all_scores) > 0:
                            print(f"  [Defense] Canary scores: min={canary_scores.min():.6f}, max={canary_scores.max():.6f}, mean={canary_scores.mean():.6f}")
                            print(f"  [Defense] All sample scores: min={all_scores.min():.6f}, max={all_scores.max():.6f}, mean={all_scores.mean():.6f}")
                            
                            # Show scores of newly filtered samples
                            newly_marked_all_scores = scores[newly_marked_indices]
                            print(f"  [Defense] Newly filtered sample scores: min={newly_marked_all_scores.min():.6f}, max={newly_marked_all_scores.max():.6f}, mean={newly_marked_all_scores.mean():.6f}")
                else:
                    if rank == 0:
                        print(f"  [Defense] Marked {n_newly_marked} samples for filtering")

            scores.fill(0)

    defense_stats = {
        'canary_indices': canary_indices_np,
        'canary_drop_epochs': canary_drop_epochs,
        'canary_drop_ratio_events': canary_drop_ratio_events,
    }
    return model, drop_mask, defense_stats


def distribute_reps(n_reps, world_size):
    """Distribute model training repetitions across GPUs"""
    reps_per_gpu = [[] for _ in range(world_size)]
    for i in range(n_reps):
        gpu_id = i % world_size
        reps_per_gpu[gpu_id].append(i)
    return reps_per_gpu


def main():
    parser = argparse.ArgumentParser(description='Audit DP-SGD using custom DP-SGD implementation (multi-canary)', allow_abbrev=False)
    
    # Check if running under torchrun (distributed mode)
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        dist.init_process_group(
            backend='nccl',
            init_method='env://'
        )
        
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        rank = int(os.environ.get('RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        
        if torch.cuda.is_available():
            device = torch.device(f'cuda:{local_rank}')
            torch.cuda.set_device(device)
            print(f'[Rank {rank}] Using device: {torch.cuda.get_device_name(local_rank)}')
        else:
            device = torch.device('cpu')
            print(f'[Rank {rank}] CUDA not available, using CPU')
    else:
        # Single GPU mode (no distributed training)
        local_rank = 0
        rank = 0
        world_size = 1
        
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            torch.cuda.set_device(device)
            print(f'Single GPU mode - Using device: {torch.cuda.get_device_name(0)}')
        else:
            device = torch.device('cpu')
            print(f'Single GPU mode - Using CPU')
    
    # Parse arguments
    parser.add_argument('--local_rank', type=int, default=0,
                        help='Local rank for distributed training')
    parser.add_argument('--data_name', type=str, default='mnist')
    parser.add_argument('--model_name', type=str, default='lr', choices=list(Models.keys()))
    parser.add_argument('--n_reps', type=int, default=200)
    parser.add_argument('--n_df', type=int, default=0)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--epsilon', type=float, default=10.0)
    parser.add_argument('--delta', type=float, default=1e-5)
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', type=str, default='parallel_results_multi_canary/')
    parser.add_argument('--fixed_init', type=str, nargs='?', default=None, const='', help='initialize all models to the same weights (if path provided, weights loaded from path (worst-case), else fix to some randomly chosen weights)')

    parser.add_argument('--holdout_audit', action='store_true')

    parser.add_argument('--batch_size', type=int, default=4000)
    parser.add_argument('--block_size', type=int, default=1024)
    parser.add_argument('--device', type=str, default=None)

    parser.add_argument('--n_canaries', type=int, default=1)
    parser.add_argument('--target_type', type=str, default='blank', choices=['blank', 'sanity_check', 'clipbkd', 'fgsm', 'mislabeled', 'gradient_space_canary'])
    parser.add_argument('--canary_pt', type=str, default=None, help='Path to a .pt file containing canaries + audit labels (overrides --target_type/--n_canaries)')
    parser.add_argument('--gradient_space_canary_pt', type=str, default=None,
                        help='Path to a .pt file containing a pre-crafted gradient space canary (dict of parameter gradients). Used when --target_type=gradient_space_canary.')
    parser.add_argument('--blank_alpha', type=float, default=0.0)

    parser.add_argument('--aug_mult', type=int, default=1)

    parser.add_argument('--defense', action='store_true')
    parser.add_argument('--defense_k', type=int, default=5)
    parser.add_argument('--defense_apply_ascent', action='store_true', default=False)
    parser.add_argument('--defense_score_norm', type=str, default='linf', choices=['linf', 'l2', 'l1'])
    parser.add_argument('--defense_score_fn', type=str, default='grad_norm')
    parser.add_argument('--defense_global_filter', action='store_true', default=False, help='Use global filtering (top-k across entire dataset) instead of per-class filtering')
    
    parser.add_argument('--fit_world_only', type=str, default=None, choices=['in', 'out'], help='just fit models in world and calculate losses')

    args = parser.parse_args()

    if args.epsilon == -1:
        args.epsilon = None

    if args.max_grad_norm == -1:
        args.max_grad_norm = None

    # Keep privacy knobs consistent: either both are set (private) or both are None (non-private).
    if args.epsilon is None or args.max_grad_norm is None:
        args.epsilon = None
        args.max_grad_norm = None

    # Device is already set above in distributed training setup
    # Only use args.device if explicitly provided and not in distributed mode
    if args.device is not None and world_size == 1:
        device = _resolve_device(args.device)

    out_folder = os.path.join(args.out, f"{args.data_name}_{args.model_name}_eps{args.epsilon}")
    if rank == 0:
        os.makedirs(out_folder, exist_ok=True)

    # Load D-
    if args.n_df == 1:
        X_out, y_out, out_dim = load_data(args.data_name, 1)
    else:
        X_out, y_out, out_dim = load_data(args.data_name, args.n_df - 1)

    y_out = y_out.long()

    canary_meta = {}
    
    # Initialize gradient_space_canary_target_class early (will be set later if gradient space canary is loaded)
    gradient_space_canary_target_class = None

    # Initialize model with SAME seed across all GPUs for fixed_init
    if rank == 0:
        print('Initializing model')
    init_model = None
    if args.fixed_init is not None:
        # Use same seed for all GPUs to ensure identical initialization
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        
        init_model = Models[args.model_name](X_out.shape, out_dim=out_dim)
        if args.model_name == 'cnn':
            xavier_init_model(init_model)
        else:
            init_wideresnet(init_model)
        
        if args.fixed_init == '':
            # Empty string: use the randomly initialized weights above
            if args.model_name == 'cnn':
                xavier_init_model(init_model)
            else:
                init_wideresnet(init_model)
        else:
            # Path provided: load pretrained weights
            init_model.load_state_dict(torch.load(args.fixed_init))
            X_out, y_out = X_out[len(X_out) // 2:], y_out[len(y_out) // 2:]
    
    # NOW set per-rank seeds for everything else (after init_model is created)
    # This ensures data loading and other operations are still independent per GPU
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)

    # Handle gradient space canary loading
    crafted_grad = None
    if args.target_type == 'gradient_space_canary':
        if args.gradient_space_canary_pt is not None:
            if not os.path.exists(args.gradient_space_canary_pt):
                raise FileNotFoundError(f"--gradient_space_canary_pt not found: {args.gradient_space_canary_pt}")
            payload = torch.load(args.gradient_space_canary_pt, map_location='cpu')
            
            # Support multiple gradient space canaries
            if isinstance(payload, dict):
                if 'gradients' in payload:
                    # Multiple gradients: list of gradient dicts
                    gradients_list = payload['gradients']
                    if not isinstance(gradients_list, list):
                        raise ValueError("'gradients' must be a list of gradient dictionaries")
                    # Move each gradient to device, filter tensors, and squeeze batch dimension
                    crafted_grad = [
                        {name: g.squeeze(0).to(device) for name, g in grad_dict.items() if torch.is_tensor(g)}
                        for grad_dict in gradients_list
                    ]
                    if rank == 0:
                        print(f"Loaded {len(crafted_grad)} gradient space canaries from {args.gradient_space_canary_pt}")
                elif 'gradient' in payload:
                    # Single gradient
                    single_grad = payload['gradient']
                    crafted_grad = {name: g.to(device) for name, g in single_grad.items() if torch.is_tensor(g)}
                    if rank == 0:
                        print(f"Loaded single gradient space canary from {args.gradient_space_canary_pt}")
                    # Extract target class if available
                    if 'target_class' in payload:
                        gradient_space_canary_target_class = payload['target_class']
                        if rank == 0:
                            print(f"  Target class: {gradient_space_canary_target_class}")
                else:
                    # Backward compatibility: direct gradient dictionary
                    crafted_grad = {name: g.to(device) for name, g in payload.items() if torch.is_tensor(g)}
                    if rank == 0:
                        print(f"Loaded gradient space canary (legacy format) from {args.gradient_space_canary_pt}")
            else:
                raise ValueError(f"Expected dict from {args.gradient_space_canary_pt}, got {type(payload)}")
        elif args.canary_pt is None:
            if rank == 0:
                print('Creating crafted gradient')
            temp_model = Models[args.model_name](X_out.shape, out_dim=out_dim).to(device)
            if args.model_name == 'cnn':
                xavier_init_model(temp_model)
            else:
                init_wideresnet(temp_model)
            crafted_grad = craft_gradient(model=temp_model, device=device)
            del temp_model
    
    # Create/load canaries
    if args.canary_pt is not None:
        X_canary, y_canary, canary_meta = _load_canaries_from_pt_dict(args.canary_pt, X_out[[0]])
        n_canaries = int(X_canary.shape[0])

        # Optional: use init model state from canary file (if compatible)
        init_state = canary_meta.get('init_model_state', None)
        if init_state is not None:
            try:
                init_model.load_state_dict(init_state)
            except Exception:
                # Don't fail hard; canary file might not include matching architecture.
                pass
    else:
        n_canaries = int(args.n_canaries)
        if n_canaries < 1:
            raise ValueError(f"--n_canaries must be >= 1, got {n_canaries}")

        if args.target_type == 'blank':
            X_canary, y_canary = _make_blank_canaries(X_out[[0]], y_out[[0]], n_canaries, args.blank_alpha)
        elif args.target_type == 'sanity_check':
            X_canary, y_canary = _make_sanity_check_canaries(X_out, y_out, n_canaries)
        elif args.target_type == 'clipbkd':
            # craft_clipbkd returns a single canary; repeat it
            X1, y1 = craft_clipbkd(X_out, init_model)
            X_canary = X1.repeat(n_canaries, *([1] * (X1.ndim - 1)))
            y_canary = y1.repeat(n_canaries)
        elif args.target_type == 'fgsm':
            if n_canaries != 1:
                raise ValueError("FGSM target generation currently supports --n_canaries=1 only")

            # Train helper model non-privately using our local training loop
            fgsm_model, _, _ = train_model_multi_canary(
                model_name=args.model_name,
                X=X_out,
                y=y_out,
                epsilon=None,
                delta=1e-5,
                max_grad_norm=None,
                n_epochs=int(args.n_epochs),
                lr=float(args.lr),
                block_size=min(int(args.block_size), int(args.batch_size)),
                batch_size=min(int(args.batch_size), len(X_out)),
                init_model=copy.deepcopy(init_model),
                out_dim=out_dim,
                aug_mult=1,
                defense=False,
                defense_k=int(args.defense_k),
                defense_apply_ascent=bool(args.defense_apply_ascent),
                defense_filter_every=1,
                defense_score_fn=str(args.defense_score_fn),
                defense_score_norm=str(args.defense_score_norm),
                defense_global_filter=bool(args.defense_global_filter),
                device=str(device),
                generator=None,
                dl_generator=None,
                num_workers=0,
                persistent_workers=False,
                canary_indices=None,
                is_gradient_space_canary=False,
                global_idx_to_grad=None,
                rank=rank,
            )

            original_X = X_out[-1].unsqueeze(0).to(device)
            original_y = y_out[-1].unsqueeze(0).to(device)
            target_class = (original_y + 1) % out_dim
            adv_X, _ = fgsm_attack(fgsm_model, original_X, target_class, epsilon=0.1, max_iter=20, alpha=0.01)
            X_canary = adv_X.detach().cpu()
            y_canary = target_class.detach().cpu()
        elif args.target_type == 'mislabeled':
            X_canary, y_canary = _make_canaries_mislabeled(
                X_out,
                y_out,
                n_canaries=n_canaries,
                out_dim=out_dim,
                seed=int(args.seed),
            )
            if rank == 0:
                y_true = y_out[-n_canaries:]
                print(f"Mislabeled canaries created:")
                print(f"  First 10 true labels: {y_true[:10].tolist()}")
                print(f"  First 10 mislabeled: {y_canary[:10].tolist()}")
                print(f"  Verification - all different: {(y_true != y_canary).all().item()}")
        elif args.target_type == 'gradient_space_canary':
            # For gradient space canary, determine number of canaries
            if isinstance(crafted_grad, list):
                n_canaries = len(crafted_grad)
            else:
                n_canaries = 1
            
            # Use the last n_canaries samples from D- as the canaries
            X_canary = X_out[-n_canaries:]
            y_canary = y_out[-n_canaries:]
            
            # Override labels if target_class is provided (single canary case)
            if gradient_space_canary_target_class is not None and n_canaries == 1:
                y_canary = torch.tensor([gradient_space_canary_target_class], dtype=torch.long)
            
            if rank == 0:
                print(f"Using {n_canaries} gradient-space canary(ies)")
        else:
            raise ValueError(f"Unknown target_type: {args.target_type}")

    if len(X_out) <= n_canaries:
        raise ValueError(f"Need len(D-) > n_canaries; got len(D-)={len(X_out)} and n_canaries={n_canaries}")

    # Define D by replacing last n_canaries samples
    X_in = torch.vstack((X_out[:-n_canaries], X_canary))
    y_in = torch.cat((y_out[:-n_canaries], y_canary.long()))
    canary_indices = np.arange(len(X_in) - n_canaries, len(X_in), dtype=np.int64)

    # Create global index to gradient mapping for gradient space canaries (once, outside training loop)
    global_idx_to_grad = None
    if args.target_type == 'gradient_space_canary' and crafted_grad is not None:
        if isinstance(crafted_grad, list):
            global_idx_to_grad = {int(canary_indices[pos]): crafted_grad[pos] for pos in range(len(canary_indices))}
        else:
            # Single gradient: use for all canaries
            global_idx_to_grad = {int(canary_idx): crafted_grad for canary_idx in canary_indices}
        if rank == 0:
            print(f"Created global index to gradient mapping for {len(global_idx_to_grad)} canaries")

    # Load test set for accuracy evaluation
    X_test, y_test, _ = load_data(args.data_name, None, split='test')
    
    # Track accuracies
    train_set_accs = []
    test_set_accs = []

    if rank == 0:
        print('Training models')
    
    # Train models and collect max canary scores
    n_reps_half = int(args.n_reps) // 2
    worlds = [args.fit_world_only] if args.fit_world_only else ['out', 'in']

    # Distribute repetitions across GPUs
    reps_per_gpu = distribute_reps(n_reps_half, world_size)
    my_reps = reps_per_gpu[rank]
    
    if rank == 0:
        print(f"Rep distribution: {[len(r) for r in reps_per_gpu]}")

    scores = {'out': [], 'in': []}

    for world in worlds:
        curr_X, curr_y = (X_out, y_out) if world == 'out' else (X_in, y_in)
        
        # Each rank trains its assigned models
        for rep_idx, rep in enumerate(my_reps):
            print(f"[Rank {rank}] Training rep {rep_idx+1}/{len(my_reps)} (global rep {rep}, world={world})")
            
            # Create unique generators for each repetition
            # Use rep (global repetition number) to ensure uniqueness across all GPUs
            rep_seed = int(args.seed) + rep * 2 + (0 if world == 'out' else 100000)
            gen = torch.Generator(device='cpu')
            gen.manual_seed(rep_seed)
            dl_gen = torch.Generator(device='cpu')
            dl_gen.manual_seed(rep_seed + 1)

            start = time.time()
            
            # For gradient space canaries, we need to save the initial model state before training
            # to compute the parameter update norm as the score
            if args.target_type == 'gradient_space_canary':
                # Create a fresh model for this repetition
                if init_model is None:
                    rep_init_model = Models[args.model_name](curr_X.shape, out_dim=out_dim)
                    if args.model_name == 'cnn':
                        xavier_init_model(rep_init_model)
                    else:
                        init_wideresnet(rep_init_model)
                else:
                    rep_init_model = copy.deepcopy(init_model)
                
                # Save initial parameters before training
                rep_init_model = rep_init_model.to(device)
                init_params_saved = {n: p.detach().clone() for n, p in rep_init_model.named_parameters()}
            else:
                rep_init_model = init_model
            
            model, _drop_mask, _defense_stats = train_model_multi_canary(
                model_name=args.model_name,
                X=curr_X,
                y=curr_y,
                epsilon=float(args.epsilon) if args.epsilon is not None else None,
                delta=float(args.delta) if args.delta is not None else None,
                max_grad_norm=float(args.max_grad_norm) if args.max_grad_norm is not None else None,
                n_epochs=int(args.n_epochs),
                lr=float(args.lr),
                block_size=min(int(args.block_size), int(args.batch_size)),
                batch_size=min(int(args.batch_size), int(curr_X.shape[0])),
                init_model=rep_init_model,
                out_dim=out_dim,
                aug_mult=int(args.aug_mult),
                defense=bool(args.defense),
                defense_k=int(args.defense_k),
                defense_apply_ascent=bool(args.defense_apply_ascent),
                defense_filter_every=1,
                defense_score_fn=str(args.defense_score_fn),
                defense_score_norm=str(args.defense_score_norm),
                defense_global_filter=bool(args.defense_global_filter),
                device=str(device),
                generator=gen,
                dl_generator=dl_gen,
                num_workers=0,
                persistent_workers=False,
                canary_indices=canary_indices if (world == 'in') else None,
                is_gradient_space_canary=(args.target_type == 'gradient_space_canary' and world == 'in'),
                global_idx_to_grad=global_idx_to_grad if (args.target_type == 'gradient_space_canary' and world == 'in') else None,
                rank=rank,
            )

            if args.target_type == 'gradient_space_canary':
                # For gradient space canary, score by L∞ norm of parameter update in both worlds
                model.eval()
                with torch.no_grad():
                    final_params = {n: p.detach().clone().to(device) for n, p in model.named_parameters()}
                    # Use the saved initial parameters from before training
                    init_params = init_params_saved
                    
                    update = {n: final_params[n] - init_params[n] for n in final_params}
                    flat_update = torch.cat([p.view(-1) for p in update.values()])
                    
                    update_norm = flat_update.norm(p=float('inf')).item()
                    max_score = float(update_norm)
                    if rank == 0:
                        print(f"[Rank {rank}] Gradient space canary score (L∞ norm of update): {max_score:.6f}")
            else:
                per_canary = _score_canaries(model, X_canary, y_canary)
                max_score = float(per_canary.max())
                if rank == 0 and rep == 0:
                    # Additional debug: check what the model actually predicts
                    model.eval()
                    with torch.no_grad():
                        dev = next(model.parameters()).device
                        logits = model(X_canary[:10].to(dev))
                        preds = logits.argmax(dim=1).cpu()
                        y_true_first10 = y_out[-n_canaries:][:10] if args.target_type == 'mislabeled' else None
                    print(f"[Rank {rank}] world={world} Scoring canaries with mislabeled labels")
                    print(f"  Per-canary scores (first 10): {per_canary[:10]}")
                    print(f"  Max score: {max_score:.6f}, Min score: {per_canary.min():.6f}")
                    if y_true_first10 is not None:
                        print(f"  True labels (first 10): {y_true_first10.tolist()}")
                        print(f"  Mislabeled (first 10): {y_canary[:10].tolist()}")
                        print(f"  Model predictions (first 10): {preds.tolist()}")
                        print(f"  Model predicts mislabel: {(preds == y_canary[:10]).sum().item()}/10")
                        print(f"  Model predicts true label: {(preds == y_true_first10).sum().item()}/10")
            
            scores[world].append(max_score)

            print(f"[Rank {rank}] world={world} rep={rep} max_canary_score={max_score:.6f} elapsed_s={time.time()-start:.2f}", flush=True)
            
            # Get test set accuracy from first 5 reps
            if rep < 5 and world == 'in':
                if len(X_out) > 0:
                    train_acc = test_model(model, X_in, y_in)
                    train_set_accs.append(train_acc)
                    print(f'[Rank {rank}] Train set acc: {train_acc:.4f}')
                test_acc = test_model(model, X_test, y_test)
                test_set_accs.append(test_acc)
                print(f'[Rank {rank}] Test set acc: {test_acc:.4f}')

    scores_in = np.asarray(scores['in'], dtype=np.float32)
    scores_out = np.asarray(scores['out'], dtype=np.float32)

    # Save per-rank results
    suffix = f'_rank{rank}' if rank > 0 else ''
    if args.fit_world_only:
        np.save(os.path.join(out_folder, f'scores_{args.fit_world_only}{suffix}.npy'), 
                scores_in if args.fit_world_only == 'in' else scores_out)
        print(f"[Rank {rank}] Saved scores for world '{args.fit_world_only}'")
    else:
        np.save(os.path.join(out_folder, f'scores_in{suffix}.npy'), scores_in)
        np.save(os.path.join(out_folder, f'scores_out{suffix}.npy'), scores_out)
        print(f"[Rank {rank}] Saved scores")
    
    # Wait for all ranks to finish
    if world_size > 1:
        dist.barrier()
    
    # Rank 0 combines results from all ranks
    emp_eps, threshold, mia_scores, mia_labels = None, None, None, None
    if rank == 0:
        print("\n[Rank 0] Combining results from all ranks...")
        combined_scores_in = []
        combined_scores_out = []
        
        for r in range(world_size):
            suffix = f'_rank{r}' if r > 0 else ''
            try:
                if not args.fit_world_only:
                    combined_scores_in.extend(np.load(f'{out_folder}/scores_in{suffix}.npy'))
                    combined_scores_out.extend(np.load(f'{out_folder}/scores_out{suffix}.npy'))
                else:
                    if args.fit_world_only == 'in':
                        combined_scores_in.extend(np.load(f'{out_folder}/scores_in{suffix}.npy'))
                    else:
                        combined_scores_out.extend(np.load(f'{out_folder}/scores_out{suffix}.npy'))
            except FileNotFoundError:
                print(f"Warning: Could not find results for rank {r}")
        
        # Save combined results
        if not args.fit_world_only:
            scores_in_combined = np.asarray(combined_scores_in, dtype=np.float32)
            scores_out_combined = np.asarray(combined_scores_out, dtype=np.float32)
            
            np.save(os.path.join(out_folder, 'scores_in.npy'), scores_in_combined)
            np.save(os.path.join(out_folder, 'scores_out.npy'), scores_out_combined)
            
            # Compute audit
            emp_eps, threshold, mia_scores, mia_labels = _audit_from_scores(
                scores_in_combined,
                scores_out_combined,
                float(args.alpha),
                float(args.delta),
                bool(args.holdout_audit),
                seed=int(args.seed),
            )
            
            np.save(os.path.join(out_folder, 'emp_eps_loss.npy'), np.asarray(emp_eps, dtype=np.float32))
            np.save(os.path.join(out_folder, 'mia_threshold.npy'), np.asarray(threshold, dtype=np.float32))
            np.save(os.path.join(out_folder, 'mia_scores.npy'), mia_scores)
            np.save(os.path.join(out_folder, 'mia_labels.npy'), mia_labels)
            
            print(f'Empirical eps: {emp_eps}')
        else:
            if args.fit_world_only == 'in':
                np.save(os.path.join(out_folder, 'scores_in.npy'), np.asarray(combined_scores_in, dtype=np.float32))
            else:
                np.save(os.path.join(out_folder, 'scores_out.npy'), np.asarray(combined_scores_out, dtype=np.float32))
            print(f"Saved combined scores for world '{args.fit_world_only}'")

    dill.dump(
        {
            'data_name': args.data_name,
            'model_name': args.model_name,
            'n_df': int(args.n_df),
            'n_canaries': int(n_canaries),
            'target_type': args.target_type,
            'epsilon': float(args.epsilon) if args.epsilon is not None else None,
            'delta': float(args.delta) if args.delta is not None else None,
            'alpha': float(args.alpha),
            'max_grad_norm': float(args.max_grad_norm) if args.max_grad_norm is not None else None,
            'batch_size': int(args.batch_size),
            'block_size': int(args.block_size),
            'n_epochs': int(args.n_epochs),
            'lr': float(args.lr),
            'seed': int(args.seed),
            'aug_mult': int(args.aug_mult),
            'holdout_audit': bool(args.holdout_audit),
            'defense': bool(args.defense),
            'defense_k': int(args.defense_k),
            'defense_score_fn': str(args.defense_score_fn),
            'defense_score_norm': str(args.defense_score_norm),
            'canary_pt': args.canary_pt,
        },
        open(os.path.join(out_folder, 'meta.dill'), 'wb'),
    )

    if rank == 0:
        if not args.fit_world_only:
            print(f"\nAUDIT RESULTS")
            print(f"Theoretical epsilon: {args.epsilon}")
            print(f"Empirical epsilon: {emp_eps}")
        else:
            print(f"\nFit world '{args.fit_world_only}' only - no audit performed")
    
    print(f"[Rank {rank}] Finished!")
    
    # Only destroy process group if we initialized it (distributed mode)
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        dist.destroy_process_group()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nInterrupted by user')
    except Exception as e:
        print(f'\nError in main: {str(e)}')
        import traceback
        traceback.print_exc()
        raise
