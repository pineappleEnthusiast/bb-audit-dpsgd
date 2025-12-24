"""
Utility functions to execute DP-SGD
"""
import torch
import torch.distributed as dist
import numpy as np
from torch.func import functional_call, vmap, grad
import matplotlib.pyplot as plt
import threading
import copy
import pdb
from opacus.grad_sample import GradSampleModule
from models.lstm import LSTM
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional

def preaugment_batch_vectorized(X, y, aug_fn, aug_mult):
    if aug_mult == 1:
        return aug_fn(X), y
    
    X_rep = X.repeat_interleave(aug_mult, dim=0)
    X_aug = aug_fn(X_rep)
    
    # Repeat labels: [B] -> [B * aug_mult]
    y_aug = y.repeat_interleave(aug_mult)
    
    return X_aug, y_aug


def average_grads_over_augmentations_optimized(ps_grads, batch_size, aug_mult):
    if aug_mult == 1:
        return ps_grads
    
    ps_grads_avg = {}
    for name, grad in ps_grads.items():
        param_dims = grad.shape[1:]
        grad_reshaped = grad.reshape(batch_size, aug_mult, *param_dims)
        ps_grads_avg[name] = grad_reshaped.mean(dim=1)
        
    return ps_grads_avg


# Drop-in replacement for your original functions
def preaugment_batch(X, y, aug_fn, aug_mult):
    """Original function signature maintained for compatibility"""
    return preaugment_batch_vectorized(X, y, aug_fn, aug_mult)


def average_grads_over_augmentations(ps_grads, batch_size, aug_mult):
    """Original function signature maintained for compatibility"""
    return average_grads_over_augmentations_optimized(ps_grads, batch_size, aug_mult)


def get_per_sample_grads(model, X, y, criterion):
    """Compute per-sample gradients"""
    # Check if model is DDP-wrapped
    is_ddp = hasattr(model, 'module')
    
    # Get model parameters, handling DDP case
    if is_ddp:
        # For DDP, we need to use the module's parameters but with the original names
        model_to_use = model.module
        # Create a mapping from original names to parameters
        param_mapping = {name.replace('module.', ''): param for name, param in model.named_parameters()}
    else:
        model_to_use = model
        param_mapping = dict(model.named_parameters())
    
    # map of parameter names : parameter values (without module prefix)
    params = {k: v.detach() for k, v in model_to_use.named_parameters()}
    # map of buffer names : buffer values (without module prefix)
    buffers = {k: v.detach() for k, v in model_to_use.named_buffers()}

    def compute_loss(params, buffers, sample, target):
        batch = sample.unsqueeze(0)
        targets = target.unsqueeze(0)
        
        # Forward pass - no no_grad() here to allow gradient computation
        predictions = functional_call(model_to_use, (params, buffers), (batch,))
        loss = criterion(predictions, targets)
        return loss
    
    # Compute gradients
    ft_compute_grad = grad(compute_loss)
    ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0, 0))
    
    # Get gradients with consistent naming (without module prefix)
    ps_grads = ft_compute_sample_grad(params, buffers, X, y)
    
    return ps_grads


def get_per_sample_grads_and_losses(model, X, y, criterion):
    """Compute per-sample gradients and the per-sample losses used to compute them."""
    # Check if model is DDP-wrapped
    is_ddp = hasattr(model, 'module')

    # Get model parameters, handling DDP case
    if is_ddp:
        model_to_use = model.module
    else:
        model_to_use = model

    # map of parameter names : parameter values (without module prefix)
    params = {k: v.detach() for k, v in model_to_use.named_parameters()}
    # map of buffer names : buffer values (without module prefix)
    buffers = {k: v.detach() for k, v in model_to_use.named_buffers()}

    def compute_loss(params, buffers, sample, target):
        batch = sample.unsqueeze(0)
        targets = target.unsqueeze(0)
        predictions = functional_call(model_to_use, (params, buffers), (batch,))
        loss = criterion(predictions, targets)
        return loss, loss

    ft_compute_grad = grad(compute_loss, has_aux=True)
    ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0, 0))

    ps_grads, ps_losses = ft_compute_sample_grad(params, buffers, X, y)
    ps_losses = ps_losses.squeeze(-1).detach()

    return ps_grads, ps_losses


def compute_per_sample_losses_from_logits(logits, y):
    # Handle both classification (B, C) and sequence modeling (B, T, C)
    if y.ndim == 2 and logits.ndim == 3:
        b, t, c = logits.shape
        token_losses = F.cross_entropy(
            logits.reshape(b * t, c),
            y.reshape(b * t),
            reduction='none'
        ).reshape(b, t)
        return token_losses.mean(dim=1)

    return F.cross_entropy(logits, y, reduction='none')


def compute_per_sample_inverse_confidence_from_logits(logits):
    # Handle both classification (B, C) and sequence modeling (B, T, C)
    if logits.ndim == 3:
        probs = logits.softmax(dim=-1)
        max_probs = probs.max(dim=-1).values
        return (1.0 - max_probs).mean(dim=1)

    probs = logits.softmax(dim=-1)
    max_probs = probs.max(dim=-1).values
    return 1.0 - max_probs


def compute_per_sample_prediction_margin_from_logits(logits, y):
    # Handle both classification (B, C) and sequence modeling (B, T, C)
    if logits.ndim == 3:
        # y: (B, T)
        probs = logits.softmax(dim=-1)
        p_true = probs.gather(dim=-1, index=y.unsqueeze(-1)).squeeze(-1)

        # max over classes excluding the true class
        mask = torch.zeros_like(probs, dtype=torch.bool)
        mask.scatter_(dim=-1, index=y.unsqueeze(-1), value=True)
        max_other = probs.masked_fill(mask, float('-inf')).max(dim=-1).values

        margin = p_true - max_other
        return margin.mean(dim=1)

    # logits: (B, C), y: (B,)
    probs = logits.softmax(dim=-1)
    p_true = probs.gather(dim=-1, index=y.unsqueeze(-1)).squeeze(-1)

    mask = torch.zeros_like(probs, dtype=torch.bool)
    mask.scatter_(dim=-1, index=y.unsqueeze(-1), value=True)
    max_other = probs.masked_fill(mask, float('-inf')).max(dim=-1).values

    return p_true - max_other


def compute_per_sample_prediction_entropy_from_logits(logits):
    # Handle both classification (B, C) and sequence modeling (B, T, C)
    if logits.ndim == 3:
        probs = logits.softmax(dim=-1)
        log_probs = torch.log(probs + 1e-12)
        ent = -(probs * log_probs).sum(dim=-1)
        return ent.mean(dim=1)

    probs = logits.softmax(dim=-1)
    log_probs = torch.log(probs + 1e-12)
    return -(probs * log_probs).sum(dim=-1)


@dataclass
class DefenseConfig:
    score_fn: str = 'grad_norm'
    score_norm: str = 'linf'
    delta_theta: Optional[torch.Tensor] = None
    theta_t_minus_theta0: Optional[torch.Tensor] = None
    prev_losses: Optional[np.ndarray] = None
    loss_hist: Optional[np.ndarray] = None
    loss_hist_pos: Optional[np.ndarray] = None
    loss_volatility_k: int = 5
    grad_norm_hist: Optional[np.ndarray] = None
    grad_norm_hist_pos: Optional[np.ndarray] = None
    grad_norm_percentile_k: int = 20
    grad_dir_hist: Optional[np.ndarray] = None
    grad_dir_hist_pos: Optional[np.ndarray] = None
    grad_dir_volatility_k: int = 5
    grad_dir_proj: Optional[torch.Tensor] = None
    rand_proj_mat: Optional[torch.Tensor] = None
    rand_proj_var_m: int = 10
    maxmin_proj_mat: Optional[torch.Tensor] = None
    maxmin_proj_k: int = 10
    grad_rank_mode: str = 'effdim'
    grad_rank_eps: float = 1e-12
    grad_accel_hist: Optional[np.ndarray] = None
    grad_accel_hist_pos: Optional[np.ndarray] = None
    grad_accel_proj: Optional[torch.Tensor] = None
    grad_jerk_hist: Optional[np.ndarray] = None
    grad_jerk_hist_pos: Optional[np.ndarray] = None
    grad_jerk_proj: Optional[torch.Tensor] = None
    dir_unique_hist: Optional[np.ndarray] = None
    alignment_proj_mat: Optional[torch.Tensor] = None
    alignment_proj_k: int = 10
    dir_unique_hist_pos: Optional[np.ndarray] = None
    dir_unique_k: int = 5
    prev_grad_dir: Optional[np.ndarray] = None
    grad_scatter_hist: Optional[np.ndarray] = None
    grad_scatter_hist_pos: Optional[np.ndarray] = None
    grad_scatter_k: int = 5


def _norm_p_from_name(score_norm: str):
    if score_norm == 'linf':
        return float('inf')
    if score_norm == 'l2':
        return 2
    if score_norm == 'l1':
        return 1
    raise ValueError(f"Unsupported defense_score_norm: {score_norm}")


def compute_defense_scores(ps_grads, ps_grads_clipped, y, defense_cfg: DefenseConfig, ps_losses=None, ps_logits=None):
    score_fn = defense_cfg.score_fn

    if score_fn == 'inv_confidence':
        if ps_logits is None:
            raise RuntimeError("defense_score_fn='inv_confidence' requires logits")
        return compute_per_sample_inverse_confidence_from_logits(ps_logits).to(dtype=torch.float32)

    if score_fn == 'prediction_margin':
        if ps_logits is None:
            raise RuntimeError("defense_score_fn='prediction_margin' requires logits")
        return compute_per_sample_prediction_margin_from_logits(ps_logits, y).to(dtype=torch.float32)

    if score_fn == 'pred_entropy':
        if ps_logits is None:
            raise RuntimeError("defense_score_fn='pred_entropy' requires logits")
        return compute_per_sample_prediction_entropy_from_logits(ps_logits).to(dtype=torch.float32)

    if score_fn in ['loss', 'loss_momentum', 'loss_volatility']:
        if ps_losses is None:
            raise RuntimeError("defense_score_fn='loss' requires per-sample losses")
        return ps_losses.to(dtype=torch.float32)

    if score_fn == 'grad_norm_x_loss':
        if ps_losses is None:
            raise RuntimeError("defense_score_fn='grad_norm_x_loss' requires per-sample losses")

        per_sample_flat_grads = torch.cat([g.view(g.shape[0], -1) for g in ps_grads_clipped.values()], dim=1)
        p = _norm_p_from_name(defense_cfg.score_norm)
        grad_norms = per_sample_flat_grads.norm(p=p, dim=1)
        return (grad_norms * ps_losses.to(device=grad_norms.device, dtype=grad_norms.dtype)).to(dtype=torch.float32)

    if score_fn == 'cos_update':
        per_sample_flat_grads = torch.cat([g.view(g.shape[0], -1) for g in ps_grads_clipped.values()], dim=1)
        delta_theta = defense_cfg.delta_theta
        if delta_theta is None:
            return torch.zeros((per_sample_flat_grads.shape[0],), device=per_sample_flat_grads.device, dtype=torch.float32)

        delta_theta = delta_theta.to(device=per_sample_flat_grads.device, dtype=per_sample_flat_grads.dtype)
        delta_norm = delta_theta.norm(2)
        if float(delta_norm) == 0.0:
            return torch.zeros((per_sample_flat_grads.shape[0],), device=per_sample_flat_grads.device, dtype=torch.float32)

        per_norms = per_sample_flat_grads.norm(2, dim=1) + 1e-12
        cos_sims = (per_sample_flat_grads @ delta_theta) / (per_norms * (delta_norm + 1e-12))
        return cos_sims.abs().to(dtype=torch.float32)

    if score_fn == 'cos_theta0':
        per_sample_flat_grads = torch.cat([g.view(g.shape[0], -1) for g in ps_grads_clipped.values()], dim=1)
        theta_t_minus_theta0 = defense_cfg.theta_t_minus_theta0
        if theta_t_minus_theta0 is None:
            return torch.zeros((per_sample_flat_grads.shape[0],), device=per_sample_flat_grads.device, dtype=torch.float32)

        theta_t_minus_theta0 = theta_t_minus_theta0.to(device=per_sample_flat_grads.device, dtype=per_sample_flat_grads.dtype)
        delta_norm = theta_t_minus_theta0.norm(2)
        if float(delta_norm) == 0.0:
            return torch.zeros((per_sample_flat_grads.shape[0],), device=per_sample_flat_grads.device, dtype=torch.float32)

        per_norms = per_sample_flat_grads.norm(2, dim=1) + 1e-12
        cos_sims = (per_sample_flat_grads @ theta_t_minus_theta0) / (per_norms * (delta_norm + 1e-12))
        return cos_sims.abs().to(dtype=torch.float32)

    if score_fn == 'fisher':
        per_sample_flat_grads = torch.cat([g.view(g.shape[0], -1) for g in ps_grads.values()], dim=1)
        return (per_sample_flat_grads ** 2).sum(dim=1).to(dtype=torch.float32)

    if score_fn == 'rand_proj_var':
        if defense_cfg.rand_proj_mat is None:
            raise ValueError("defense_cfg.rand_proj_mat must be provided when score_fn='rand_proj_var'")

        per_sample_flat_grads = torch.cat([g.view(g.shape[0], -1) for g in ps_grads_clipped.values()], dim=1)
        proj_mat = defense_cfg.rand_proj_mat.to(device=per_sample_flat_grads.device, dtype=per_sample_flat_grads.dtype)
        projections = per_sample_flat_grads @ proj_mat
        mean_abs = projections.abs().mean(dim=1)
        std = projections.std(dim=1, unbiased=False)
        return (mean_abs * std).to(dtype=torch.float32)

    if score_fn == 'maxmin_proj_ratio':
        if defense_cfg.maxmin_proj_mat is None:
            raise ValueError("defense_cfg.maxmin_proj_mat must be provided when score_fn='maxmin_proj_ratio'")

        per_sample_flat_grads = torch.cat([g.view(g.shape[0], -1) for g in ps_grads_clipped.values()], dim=1)
        proj_mat = defense_cfg.maxmin_proj_mat.to(device=per_sample_flat_grads.device, dtype=per_sample_flat_grads.dtype)
        projections = (per_sample_flat_grads @ proj_mat).abs()
        max_proj = projections.max(dim=1).values
        min_proj = projections.min(dim=1).values
        return (max_proj / (min_proj + 1e-12)).to(dtype=torch.float32)

    if score_fn == 'gradient_rank':
        per_sample_flat_grads = torch.cat([g.view(g.shape[0], -1) for g in ps_grads_clipped.values()], dim=1)
        d = float(per_sample_flat_grads.shape[1])
        eps = float(defense_cfg.grad_rank_eps)

        abs_g = per_sample_flat_grads.abs()
        l1 = abs_g.sum(dim=1)
        l2 = per_sample_flat_grads.norm(2, dim=1)

        mode = str(defense_cfg.grad_rank_mode)
        if mode == 'effdim':
            effdim = (l1 / (l2 + eps)) ** 2
            return (effdim / (d + eps)).to(dtype=torch.float32)

        if mode == 'entropy':
            probs = abs_g / (l1[:, None] + eps)
            entropy = -(probs * (probs + eps).log()).sum(dim=1)
            return (l2 * entropy).to(dtype=torch.float32)

        raise ValueError(f"Unsupported grad_rank_mode: {mode}")

    if score_fn == 'alignment_with_rand_proj':
        if defense_cfg.alignment_proj_mat is None:
            raise ValueError("defense_cfg.alignment_proj_mat must be provided when score_fn='alignment_with_rand_proj'")

        per_sample_flat_grads = torch.cat([g.view(g.shape[0], -1) for g in ps_grads_clipped.values()], dim=1)
        proj_mat = defense_cfg.alignment_proj_mat.to(device=per_sample_flat_grads.device, dtype=per_sample_flat_grads.dtype)
        
        grad_norms = per_sample_flat_grads.norm(2, dim=1, keepdim=True) + 1e-12
        normalized_grads = per_sample_flat_grads / grad_norms
        
        cos_sims = normalized_grads @ proj_mat
        alignment_std = cos_sims.std(dim=1, unbiased=False)
        return alignment_std.to(dtype=torch.float32)

    if score_fn == 'gradient_sparsity':
        per_sample_flat_grads = torch.cat([g.view(g.shape[0], -1) for g in ps_grads_clipped.values()], dim=1)
        l1_norm = per_sample_flat_grads.abs().sum(dim=1)
        l2_norm = per_sample_flat_grads.norm(2, dim=1)
        return (l1_norm / (l2_norm + 1e-12)).to(dtype=torch.float32)

    if score_fn == 'gradient_kurtosis':
        per_sample_flat_grads = torch.cat([g.view(g.shape[0], -1) for g in ps_grads_clipped.values()], dim=1)
        mean_g = per_sample_flat_grads.mean(dim=1, keepdim=True)
        std_g = per_sample_flat_grads.std(dim=1, keepdim=True, unbiased=False)
        normalized = (per_sample_flat_grads - mean_g) / (std_g + 1e-12)
        kurtosis = (normalized ** 4).mean(dim=1)
        return kurtosis.to(dtype=torch.float32)

    if score_fn == 'grad_dir_change_rate':
        per_sample_flat_grads = torch.cat([g.view(g.shape[0], -1) for g in ps_grads_clipped.values()], dim=1)
        grad_norms = per_sample_flat_grads.norm(2, dim=1, keepdim=True) + 1e-12
        normalized_grads = per_sample_flat_grads / grad_norms
        
        if defense_cfg.prev_grad_dir is None:
            defense_cfg.prev_grad_dir = normalized_grads.detach().cpu().numpy()
            return torch.zeros((per_sample_flat_grads.shape[0],), device=per_sample_flat_grads.device, dtype=torch.float32)
        
        prev_dirs = torch.from_numpy(defense_cfg.prev_grad_dir).to(device=normalized_grads.device, dtype=normalized_grads.dtype)
        cos_sims = (normalized_grads * prev_dirs).sum(dim=1)
        direction_change = 1.0 - cos_sims
        
        defense_cfg.prev_grad_dir = normalized_grads.detach().cpu().numpy()
        return direction_change.to(dtype=torch.float32)

    if score_fn == 'norm_x_trajectory_orth':
        per_sample_flat_grads = torch.cat([g.view(g.shape[0], -1) for g in ps_grads_clipped.values()], dim=1)
        theta_t_minus_theta0 = defense_cfg.theta_t_minus_theta0
        if theta_t_minus_theta0 is None:
            return torch.zeros((per_sample_flat_grads.shape[0],), device=per_sample_flat_grads.device, dtype=torch.float32)
        
        theta_t_minus_theta0 = theta_t_minus_theta0.to(device=per_sample_flat_grads.device, dtype=per_sample_flat_grads.dtype)
        trajectory_norm = theta_t_minus_theta0.norm(2)
        if float(trajectory_norm) == 0.0:
            return torch.zeros((per_sample_flat_grads.shape[0],), device=per_sample_flat_grads.device, dtype=torch.float32)
        
        overall_direction = theta_t_minus_theta0 / (trajectory_norm + 1e-12)
        grad_norms = per_sample_flat_grads.norm(2, dim=1)
        normalized_grads = per_sample_flat_grads / (grad_norms[:, None] + 1e-12)
        
        cos_sims = (normalized_grads @ overall_direction).abs()
        orthogonality = 1.0 - cos_sims
        
        return (grad_norms * orthogonality).to(dtype=torch.float32)

    if score_fn == 'gradient_scatter':
        per_sample_flat_grads = torch.cat([g.view(g.shape[0], -1) for g in ps_grads_clipped.values()], dim=1)
        n_samples = per_sample_flat_grads.shape[0]
        grad_dim = per_sample_flat_grads.shape[1]
        k = int(defense_cfg.grad_scatter_k)
        
        if defense_cfg.grad_scatter_hist is None:
            defense_cfg.grad_scatter_hist = np.zeros((n_samples, k, grad_dim), dtype=np.float32)
            defense_cfg.grad_scatter_hist_pos = np.zeros(n_samples, dtype=np.int32)
        
        curr_grads_np = per_sample_flat_grads.detach().cpu().numpy()
        scatter_scores = np.zeros(n_samples, dtype=np.float32)
        
        for i in range(n_samples):
            pos = int(defense_cfg.grad_scatter_hist_pos[i])
            defense_cfg.grad_scatter_hist[i, pos] = curr_grads_np[i]
            defense_cfg.grad_scatter_hist_pos[i] = (pos + 1) % k
            
            recent_grads = defense_cfg.grad_scatter_hist[i]
            centroid = recent_grads.mean(axis=0)
            scatter = ((recent_grads - centroid) ** 2).sum(axis=1).mean()
            scatter_scores[i] = scatter
        
        return torch.from_numpy(scatter_scores).to(device=per_sample_flat_grads.device, dtype=torch.float32)

    per_sample_flat_grads = torch.cat([g.view(g.shape[0], -1) for g in ps_grads.values()], dim=1)
    p = _norm_p_from_name(defense_cfg.score_norm)
    return per_sample_flat_grads.norm(p, dim=1).to(dtype=torch.float32)

def get_per_sample_grad_norms(per_sample_grads):
    """Compute L2 norms of per-sample gradients"""
    return torch.vstack([
        curr_grad.flatten(start_dim=1).norm(2, dim=1)
        for curr_grad in per_sample_grads.values()
    ]).norm(2, dim=0)


def clip_per_sample_grads(per_sample_grads, max_grad_norm):
    """Clip per-sample gradients to clipping norm"""
    ps_grad_norms = get_per_sample_grad_norms(per_sample_grads)

    ps_grad_scales = 1 / torch.maximum(
        torch.ones_like(ps_grad_norms),
        ps_grad_norms / max_grad_norm
    )

    ps_grads_clipped = {
        # broadcast
        name: curr_grad * ps_grad_scales[(...,) + (None,) * (curr_grad.dim() - 1)]
        for name, curr_grad in per_sample_grads.items() 
    }

    ps_grad_norms_clipped = get_per_sample_grad_norms(ps_grads_clipped)

    return ps_grads_clipped, { 'before': ps_grad_norms.cpu().numpy(), 'after': ps_grad_norms_clipped.cpu().numpy() }

def _get_per_sample_grads(model, X, y, criterion):
    model.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    ps_grads = {name: param.grad_sample for name, param in model.named_parameters()}
    return ps_grads

def clip_and_accum_grads_block(model, X, y, optimizer, criterion, max_grad_norm, device='cuda', aug_fn=None, aug_mult=1, 
                             is_gradient_space_canary=False, crafted_gradient=None, canary_local_idx=None, curr_gradient_ascent_indices=None, defense_cfg: Optional[DefenseConfig] = None, defense_score_norm='linf', defense_score_fn='grad_norm', delta_theta=None, theta_t_minus_theta0=None, defense_apply_ascent=True):
    """
    Add aug_fn and aug_mult params to support augmentation multiplicity outside vmap.

    If aug_mult > 1, apply augmentation multiplicity outside, then average grads.
    """
    optimizer.zero_grad()
    
    # Check if model is DDP-wrapped
    is_ddp = hasattr(model, 'module')
    
    # Get the actual model (unwrapped if DDP)
    model_to_use = model.module if is_ddp else model
    
    # Get parameter names without 'module.' prefix
    param_names = [name.replace('module.', '') for name in model.state_dict().keys() 
                  if not name.startswith('_forward_hooks') and not name.startswith('_backward_hooks')]
    
    if defense_cfg is None:
        defense_cfg = DefenseConfig(
            score_fn=defense_score_fn,
            score_norm=defense_score_norm,
            delta_theta=delta_theta,
            theta_t_minus_theta0=theta_t_minus_theta0
        )

    ps_losses = None
    ps_logits = None
    if len(X) == 0:
        # Initialize zero gradients with correct names
        ps_grads = {name: torch.zeros_like(param).unsqueeze(dim=0) 
                   for name, param in model_to_use.named_parameters()}
    else:
        # Compute per-sample gradients
        # Pre-augment outside vmap if aug_mult > 1
        if aug_mult > 1 and aug_fn is not None:
            X_aug, y_aug = preaugment_batch(X, y, aug_fn, aug_mult)
            X_aug = X_aug.to(device)
            y_aug = y_aug.to(device)

            if defense_cfg.score_fn == 'inv_confidence':
                logits = model(X_aug)
                aug_inv_conf = compute_per_sample_inverse_confidence_from_logits(logits)
                ps_logits = logits
                ps_losses = None

            if defense_cfg.score_fn == 'prediction_margin':
                logits = model(X_aug)
                aug_margin = compute_per_sample_prediction_margin_from_logits(logits, y_aug)
                ps_logits = logits
                ps_losses = None

            if defense_cfg.score_fn == 'pred_entropy':
                logits = model(X_aug)
                aug_entropy = compute_per_sample_prediction_entropy_from_logits(logits)
                ps_logits = logits
                ps_losses = None

            if isinstance(model_to_use, LSTM):
                ps_grads = _get_per_sample_grads(GradSampleModule(model), X_aug, y_aug, criterion)
                if defense_cfg.score_fn in ['loss', 'loss_momentum', 'loss_volatility', 'grad_norm_x_loss']:
                    logits = model(X_aug)
                    aug_losses = compute_per_sample_losses_from_logits(logits, y_aug)
                    ps_losses = aug_losses.reshape(len(X), aug_mult).mean(dim=1).detach()
            else:
                if defense_cfg.score_fn in ['loss', 'loss_momentum', 'loss_volatility', 'grad_norm_x_loss']:
                    ps_grads, aug_losses = get_per_sample_grads_and_losses(model, X_aug, y_aug, criterion)
                    ps_losses = aug_losses.reshape(len(X), aug_mult).mean(dim=1).detach()
                else:
                    ps_grads = get_per_sample_grads(model, X_aug, y_aug, criterion)

            if defense_cfg.score_fn == 'inv_confidence':
                ps_losses = None
                ps_logits = None
                ps_inv_conf = aug_inv_conf.reshape(len(X), aug_mult).mean(dim=1).detach()
                ps_margin = None
                ps_entropy = None
            elif defense_cfg.score_fn == 'prediction_margin':
                ps_losses = None
                ps_logits = None
                ps_margin = aug_margin.reshape(len(X), aug_mult).mean(dim=1).detach()
                ps_inv_conf = None
                ps_entropy = None
            elif defense_cfg.score_fn == 'pred_entropy':
                ps_losses = None
                ps_logits = None
                ps_entropy = aug_entropy.reshape(len(X), aug_mult).mean(dim=1).detach()
                ps_inv_conf = None
                ps_margin = None
            else:
                ps_inv_conf = None
                ps_margin = None
                ps_entropy = None

            ps_grads = average_grads_over_augmentations(ps_grads, batch_size=len(X), aug_mult=aug_mult)
        else:
            X = X.to(device)
            y = y.to(device)

            if defense_cfg.score_fn == 'inv_confidence':
                logits = model(X)
                ps_inv_conf = compute_per_sample_inverse_confidence_from_logits(logits).detach()
            else:
                ps_inv_conf = None

            if defense_cfg.score_fn == 'prediction_margin':
                logits = model(X)
                ps_margin = compute_per_sample_prediction_margin_from_logits(logits, y).detach()
            else:
                ps_margin = None

            if defense_cfg.score_fn == 'pred_entropy':
                logits = model(X)
                ps_entropy = compute_per_sample_prediction_entropy_from_logits(logits).detach()
            else:
                ps_entropy = None

            if isinstance(model_to_use, LSTM):
                ps_grads = _get_per_sample_grads(GradSampleModule(model), X, y, criterion)
                if defense_cfg.score_fn in ['loss', 'loss_momentum', 'loss_volatility', 'grad_norm_x_loss']:
                    logits = model(X)
                    ps_losses = compute_per_sample_losses_from_logits(logits, y).detach()
            else:
                if defense_cfg.score_fn in ['loss', 'loss_momentum', 'loss_volatility', 'grad_norm_x_loss']:
                    ps_grads, ps_losses = get_per_sample_grads_and_losses(model, X, y, criterion)
                else:
                    ps_grads = get_per_sample_grads(model, X, y, criterion)
        
        # Apply gradient-space audit after getting the gradients but before clipping
        if is_gradient_space_canary:
            # For the last sample in the block, replace its gradient with a crafted one
            for name in ps_grads.keys():
                # Replace the last sample's gradient with the crafted one
                ps_grads[name][canary_local_idx] = crafted_gradient[name]
            
    if max_grad_norm is not None:
        ps_grads_clipped, _ = clip_per_sample_grads(ps_grads, max_grad_norm)
    else:
        ps_grads_clipped = ps_grads

    # last_layer_name = list(model.net.named_modules())[-1][0]
    # last_w_name = 'net.' + last_layer_name + '.weight'
    # last_b_name = 'net.' + last_layer_name + '.bias'

    if defense_cfg.score_fn == 'inv_confidence':
        scores_block = ps_inv_conf.to(dtype=torch.float32)
    elif defense_cfg.score_fn == 'prediction_margin':
        scores_block = ps_margin.to(dtype=torch.float32)
    elif defense_cfg.score_fn == 'pred_entropy':
        scores_block = ps_entropy.to(dtype=torch.float32)
    elif defense_cfg.score_fn == 'norm_x_dir_uniqueness':
        p = _norm_p_from_name(defense_cfg.score_norm)
        per_sample_flat_grads = torch.cat([g.view(g.shape[0], -1) for g in ps_grads_clipped.values()], dim=1)
        scores_block = per_sample_flat_grads.norm(p=p, dim=1).to(dtype=torch.float32)
    else:
        scores_block = compute_defense_scores(ps_grads, ps_grads_clipped, y, defense_cfg, ps_losses=ps_losses, ps_logits=ps_logits)

    aux_embeds_block = None
    if defense_cfg is not None and defense_cfg.score_fn in {'grad_dir_volatility', 'norm_x_dir_uniqueness'}:
        if defense_cfg.grad_dir_proj is None:
            raise ValueError("defense_cfg.grad_dir_proj must be provided when score_fn='grad_dir_volatility' or 'norm_x_dir_uniqueness'")

        per_sample_flat_grads = torch.cat([g.view(g.shape[0], -1) for g in ps_grads_clipped.values()], dim=1)
        per_sample_flat_grads = per_sample_flat_grads / (per_sample_flat_grads.norm(2, dim=1, keepdim=True) + 1e-12)

        proj = defense_cfg.grad_dir_proj.to(device=per_sample_flat_grads.device, dtype=per_sample_flat_grads.dtype)
        embeds = per_sample_flat_grads @ proj
        embeds = embeds / (embeds.norm(2, dim=1, keepdim=True) + 1e-12)
        aux_embeds_block = embeds.detach().cpu().numpy().astype(np.float32, copy=False)

    if defense_cfg is not None and defense_cfg.score_fn == 'grad_accel':
        if defense_cfg.grad_accel_proj is None:
            raise ValueError("defense_cfg.grad_accel_proj must be provided when score_fn='grad_accel'")

        per_sample_flat_grads = torch.cat([g.view(g.shape[0], -1) for g in ps_grads_clipped.values()], dim=1)
        proj = defense_cfg.grad_accel_proj.to(device=per_sample_flat_grads.device, dtype=per_sample_flat_grads.dtype)
        embeds = per_sample_flat_grads @ proj
        aux_embeds_block = embeds.detach().cpu().numpy().astype(np.float32, copy=False)

    if defense_cfg is not None and defense_cfg.score_fn == 'grad_jerk':
        if defense_cfg.grad_jerk_proj is None:
            raise ValueError("defense_cfg.grad_jerk_proj must be provided when score_fn='grad_jerk'")

        per_sample_flat_grads = torch.cat([g.view(g.shape[0], -1) for g in ps_grads_clipped.values()], dim=1)
        proj = defense_cfg.grad_jerk_proj.to(device=per_sample_flat_grads.device, dtype=per_sample_flat_grads.dtype)
        embeds = per_sample_flat_grads @ proj
        aux_embeds_block = embeds.detach().cpu().numpy().astype(np.float32, copy=False)

    # flat_last_weights = ps_grads[last_w_name].flatten(start_dim=1)
    # last_biases = ps_grads[last_b_name]
    # last_layer_grads = torch.cat((flat_last_weights, last_biases), dim=1)
    
    # all_norms = torch.zeros_like(y, dtype=torch.float32)
    # for k in range(10):
    #     # center each class
    #     k_last_layer_grads = last_layer_grads[y == k]
    #     centered_k_last_layer_grads = k_last_layer_grads - k_last_layer_grads.mean(dim=0, keepdim=True)
    #     # take norm of each class
    #     centered_k_last_layer_norms = centered_k_last_layer_grads.norm(2, dim=1)
    #     all_norms[y == k] = centered_k_last_layer_norms

    # Apply gradient ascent if enabled
    if defense_apply_ascent:
        for name in ps_grads_clipped:
            ps_grads_clipped[name][curr_gradient_ascent_indices] *= -1


    with torch.no_grad():
        accum_grad_block = {name: grad.sum(dim=0) for name, grad in ps_grads_clipped.items()}

    return accum_grad_block, None, scores_block.cpu().numpy(), aux_embeds_block





def clip_and_accum_grads(model, X, y, optimizer, criterion, max_grad_norm,
                         block_size=1024, scores=None, device='cuda',
                         global_indices=None, aug_mult: int = 1, aug_fn=None,
                         world_size=1, rank=0, batch_size=None, drop_mask=None,
                         is_gradient_space_canary=False, crafted_gradient=None, defense_cfg: Optional[DefenseConfig] = None, defense_score_norm='linf', defense_score_fn='grad_norm', delta_theta=None, theta_t_minus_theta0=None, defense_apply_ascent=True):

    if scores is None:
        raise ValueError("scores array must be provided")
    
    if drop_mask is not None and len(drop_mask) != len(X):
        raise ValueError(f"drop_mask length ({len(drop_mask)}) must match X length ({len(X)})")
    
    batch_size_in = len(X)

    # Get indices of non-dropped samples
    # TODO: bug is here
    active_indices = (torch.tensor(drop_mask, device=device) != 2)

    gradient_ascent_indices = torch.tensor(drop_mask, device=device)[active_indices] == 1

    # Filter out dropped samples
    X = X[active_indices]
    y = y[active_indices]
    global_indices = global_indices[active_indices]
    
    # Check if the canary is in this batch and we should apply gradient space canary
    apply_gradient_space_canary = is_gradient_space_canary and (global_indices == (len(scores) - 1)).any()
    
    if len(X) == 0:
        return None, scores
    
    # Process in blocks for memory efficiency
    accum_grad = None
    n_samples = len(X)
    
    for i in range(0, n_samples, block_size):
        # Get current block
        idx_block = slice(i, min(i + block_size, n_samples))
        curr_X = X[idx_block]
        curr_y = y[idx_block]
        curr_global_indices = global_indices[idx_block]

        curr_gradient_ascent_indices = gradient_ascent_indices[idx_block]
        
        # Skip if no samples in this block
        if len(curr_X) == 0:
            continue
            
        # Check if this block contains the last sample (canary)
        block_contains_canary = apply_gradient_space_canary and (curr_global_indices == (len(scores) - 1)).any()

        # Get the local index of the last sample in the current block
        if block_contains_canary:
            last_sample_local_idx = (curr_global_indices == (len(scores) - 1)).nonzero()[0].item()
        else:
            last_sample_local_idx = None
        
        # Compute per-block gradients with clipping
        accum_grad_block, _, last_layer_norms, dir_embeds_block = clip_and_accum_grads_block(
            model, curr_X, curr_y, optimizer, criterion, max_grad_norm,
            device=device, aug_mult=aug_mult, aug_fn=aug_fn,
            is_gradient_space_canary=block_contains_canary,
            crafted_gradient=crafted_gradient,
            canary_local_idx=last_sample_local_idx,
            curr_gradient_ascent_indices=curr_gradient_ascent_indices,
            defense_cfg=defense_cfg,
            defense_score_norm=defense_score_norm,
            defense_score_fn=defense_score_fn,
            delta_theta=delta_theta,
            theta_t_minus_theta0=theta_t_minus_theta0,
            defense_apply_ascent=defense_apply_ascent
        )

        # Accumulate gradients
        if accum_grad is None:
            accum_grad = accum_grad_block
        else:
            with torch.no_grad():
                for name in accum_grad:
                    accum_grad[name] += accum_grad_block[name]
        
        # Update scores for this block
        if defense_cfg is not None and defense_cfg.score_fn == 'loss_momentum':
            if defense_cfg.prev_losses is None:
                raise ValueError("defense_cfg.prev_losses must be provided when score_fn='loss_momentum'")

            curr_idx_np = curr_global_indices.detach().cpu().numpy()
            curr_losses = last_layer_norms.astype(np.float32, copy=False)
            prev_losses = defense_cfg.prev_losses[curr_idx_np]

            valid_prev = ~np.isnan(prev_losses)
            momentum = np.zeros_like(curr_losses, dtype=np.float32)
            momentum[valid_prev] = np.abs(curr_losses[valid_prev] - prev_losses[valid_prev])

            scores[curr_idx_np] = momentum
            defense_cfg.prev_losses[curr_idx_np] = curr_losses
        elif defense_cfg is not None and defense_cfg.score_fn == 'loss_volatility':
            if defense_cfg.loss_hist is None or defense_cfg.loss_hist_pos is None:
                raise ValueError("defense_cfg.loss_hist and defense_cfg.loss_hist_pos must be provided when score_fn='loss_volatility'")

            k = int(defense_cfg.loss_volatility_k)
            if k <= 0:
                raise ValueError(f"loss_volatility_k must be > 0, got {k}")
            if defense_cfg.loss_hist.shape[1] != k:
                raise ValueError(f"defense_cfg.loss_hist has shape {defense_cfg.loss_hist.shape}, expected second dim == {k}")

            curr_idx_np = curr_global_indices.detach().cpu().numpy()
            curr_losses = last_layer_norms.astype(np.float32, copy=False)

            pos = defense_cfg.loss_hist_pos[curr_idx_np].astype(np.int64, copy=False)
            defense_cfg.loss_hist[curr_idx_np, pos] = curr_losses
            defense_cfg.loss_hist_pos[curr_idx_np] = (pos + 1) % k

            volatility = np.nanstd(defense_cfg.loss_hist[curr_idx_np], axis=1).astype(np.float32, copy=False)
            scores[curr_idx_np] = volatility
        elif defense_cfg is not None and defense_cfg.score_fn == 'grad_norm_percentile':
            if defense_cfg.grad_norm_hist is None or defense_cfg.grad_norm_hist_pos is None:
                raise ValueError("defense_cfg.grad_norm_hist and defense_cfg.grad_norm_hist_pos must be provided when score_fn='grad_norm_percentile'")

            k = int(defense_cfg.grad_norm_percentile_k)
            if k <= 0:
                raise ValueError(f"grad_norm_percentile_k must be > 0, got {k}")
            if defense_cfg.grad_norm_hist.shape[1] != k:
                raise ValueError(f"defense_cfg.grad_norm_hist has shape {defense_cfg.grad_norm_hist.shape}, expected second dim == {k}")

            curr_idx_np = curr_global_indices.detach().cpu().numpy()
            curr_norms = last_layer_norms.astype(np.float32, copy=False)

            hist = defense_cfg.grad_norm_hist[curr_idx_np]
            valid = ~np.isnan(hist)
            counts = valid.sum(axis=1).astype(np.int32, copy=False)

            # Percentile rank in [0, 1]. If no history yet, define percentile as 0.
            leq = (hist <= curr_norms[:, None]) & valid
            pct = np.zeros_like(curr_norms, dtype=np.float32)
            nonzero = counts > 0
            pct[nonzero] = (leq[nonzero].sum(axis=1) / counts[nonzero]).astype(np.float32, copy=False)
            scores[curr_idx_np] = pct

            pos = defense_cfg.grad_norm_hist_pos[curr_idx_np].astype(np.int64, copy=False)
            defense_cfg.grad_norm_hist[curr_idx_np, pos] = curr_norms
            defense_cfg.grad_norm_hist_pos[curr_idx_np] = (pos + 1) % k
        elif defense_cfg is not None and defense_cfg.score_fn == 'grad_dir_volatility':
            if defense_cfg.grad_dir_hist is None or defense_cfg.grad_dir_hist_pos is None:
                raise ValueError("defense_cfg.grad_dir_hist and defense_cfg.grad_dir_hist_pos must be provided when score_fn='grad_dir_volatility'")
            if dir_embeds_block is None:
                raise RuntimeError("dir_embeds_block must be returned from clip_and_accum_grads_block when score_fn='grad_dir_volatility'")

            k = int(defense_cfg.grad_dir_volatility_k)
            if k <= 0:
                raise ValueError(f"grad_dir_volatility_k must be > 0, got {k}")
            if defense_cfg.grad_dir_hist.shape[1] != k:
                raise ValueError(f"defense_cfg.grad_dir_hist has shape {defense_cfg.grad_dir_hist.shape}, expected second dim == {k}")

            curr_idx_np = curr_global_indices.detach().cpu().numpy()
            curr_dirs = dir_embeds_block.astype(np.float32, copy=False)

            hist = defense_cfg.grad_dir_hist[curr_idx_np]
            valid = ~np.isnan(hist[..., 0])
            counts = valid.sum(axis=1).astype(np.int32, copy=False)

            # cos_sim between current dir and each historical dir (dirs are already unit norm)
            dots = (hist * curr_dirs[:, None, :]).sum(axis=2)
            diffs = 1.0 - dots

            score = np.zeros((curr_dirs.shape[0],), dtype=np.float32)
            nonzero = counts > 0
            if np.any(nonzero):
                diffs_masked = np.where(valid, diffs, np.nan)
                score[nonzero] = np.nanmean(diffs_masked[nonzero], axis=1).astype(np.float32, copy=False)
            scores[curr_idx_np] = score

            pos = defense_cfg.grad_dir_hist_pos[curr_idx_np].astype(np.int64, copy=False)
            defense_cfg.grad_dir_hist[curr_idx_np, pos, :] = curr_dirs
            defense_cfg.grad_dir_hist_pos[curr_idx_np] = (pos + 1) % k
        elif defense_cfg is not None and defense_cfg.score_fn == 'grad_accel':
            if defense_cfg.grad_accel_hist is None or defense_cfg.grad_accel_hist_pos is None:
                raise ValueError("defense_cfg.grad_accel_hist and defense_cfg.grad_accel_hist_pos must be provided when score_fn='grad_accel'")
            if dir_embeds_block is None:
                raise RuntimeError("Projected gradients must be returned from clip_and_accum_grads_block when score_fn='grad_accel'")

            curr_idx_np = curr_global_indices.detach().cpu().numpy()
            curr_embeds = dir_embeds_block.astype(np.float32, copy=False)

            hist = defense_cfg.grad_accel_hist[curr_idx_np]
            pos = defense_cfg.grad_accel_hist_pos[curr_idx_np].astype(np.int64, copy=False)

            prev1 = hist[np.arange(hist.shape[0]), (pos - 1) % 3, :]
            prev2 = hist[np.arange(hist.shape[0]), (pos - 2) % 3, :]
            valid = (~np.isnan(prev1).any(axis=1)) & (~np.isnan(prev2).any(axis=1))

            accel = np.zeros_like(curr_embeds, dtype=np.float32)
            accel[valid] = curr_embeds[valid] - 2.0 * prev1[valid] + prev2[valid]
            scores[curr_idx_np] = np.linalg.norm(accel, axis=1).astype(np.float32, copy=False)

            defense_cfg.grad_accel_hist[curr_idx_np, pos, :] = curr_embeds
            defense_cfg.grad_accel_hist_pos[curr_idx_np] = (pos + 1) % 3
        elif defense_cfg is not None and defense_cfg.score_fn == 'grad_jerk':
            if defense_cfg.grad_jerk_hist is None or defense_cfg.grad_jerk_hist_pos is None:
                raise ValueError("defense_cfg.grad_jerk_hist and defense_cfg.grad_jerk_hist_pos must be provided when score_fn='grad_jerk'")
            if dir_embeds_block is None:
                raise RuntimeError("Projected gradients must be returned from clip_and_accum_grads_block when score_fn='grad_jerk'")

            curr_idx_np = curr_global_indices.detach().cpu().numpy()
            curr_embeds = dir_embeds_block.astype(np.float32, copy=False)

            hist = defense_cfg.grad_jerk_hist[curr_idx_np]
            pos = defense_cfg.grad_jerk_hist_pos[curr_idx_np].astype(np.int64, copy=False)

            prev1 = hist[np.arange(hist.shape[0]), (pos - 1) % 4, :]
            prev2 = hist[np.arange(hist.shape[0]), (pos - 2) % 4, :]
            prev3 = hist[np.arange(hist.shape[0]), (pos - 3) % 4, :]
            valid = (~np.isnan(prev1).any(axis=1)) & (~np.isnan(prev2).any(axis=1)) & (~np.isnan(prev3).any(axis=1))

            jerk = np.zeros_like(curr_embeds, dtype=np.float32)
            jerk[valid] = curr_embeds[valid] - 3.0 * prev1[valid] + 3.0 * prev2[valid] - prev3[valid]
            scores[curr_idx_np] = np.linalg.norm(jerk, axis=1).astype(np.float32, copy=False)

            defense_cfg.grad_jerk_hist[curr_idx_np, pos, :] = curr_embeds
            defense_cfg.grad_jerk_hist_pos[curr_idx_np] = (pos + 1) % 4
        elif defense_cfg is not None and defense_cfg.score_fn == 'norm_x_dir_uniqueness':
            if defense_cfg.dir_unique_hist is None or defense_cfg.dir_unique_hist_pos is None:
                raise ValueError("defense_cfg.dir_unique_hist and defense_cfg.dir_unique_hist_pos must be provided when score_fn='norm_x_dir_uniqueness'")
            if dir_embeds_block is None:
                raise RuntimeError("Projected directions must be returned from clip_and_accum_grads_block when score_fn='norm_x_dir_uniqueness'")

            curr_idx_np = curr_global_indices.detach().cpu().numpy()
            curr_dirs = dir_embeds_block.astype(np.float32, copy=False)

            # Magnitude of clipped per-sample gradients (computed in clip_and_accum_grads_block)
            magnitude = last_layer_norms.astype(np.float32, copy=False)

            k = int(defense_cfg.dir_unique_k)
            hist = defense_cfg.dir_unique_hist[curr_idx_np]
            pos = defense_cfg.dir_unique_hist_pos[curr_idx_np].astype(np.int64, copy=False)

            # Gather previous K directions in recency order (t-1..t-K)
            past_dirs = []
            for j in range(1, k + 1):
                past_dirs.append(hist[np.arange(hist.shape[0]), (pos - j) % k, :])
            past_dirs = np.stack(past_dirs, axis=1)  # (B, K, D)

            valid = ~np.isnan(past_dirs).any(axis=2)  # (B, K)
            # Since directions are normalized, cos_sim is dot product
            cos = np.einsum('bd,bkd->bk', curr_dirs, past_dirs)
            cos = np.where(valid, cos, np.nan)

            # std over available past cos sims (need at least 2 to have nonzero std)
            counts = np.sum(~np.isnan(cos), axis=1)
            dir_vol = np.zeros((curr_dirs.shape[0],), dtype=np.float32)
            enough = counts >= 2
            if np.any(enough):
                dir_vol[enough] = np.nanstd(cos[enough], axis=1).astype(np.float32, copy=False)

            scores[curr_idx_np] = magnitude * dir_vol

            defense_cfg.dir_unique_hist[curr_idx_np, pos, :] = curr_dirs
            defense_cfg.dir_unique_hist_pos[curr_idx_np] = (pos + 1) % k
        else:
            scores[curr_global_indices.cpu().numpy()] = last_layer_norms

    if accum_grad is not None and batch_size_in > 0:
        with torch.no_grad():
            for name in accum_grad:
                accum_grad[name] = accum_grad[name] / float(batch_size_in)

    return accum_grad, scores


