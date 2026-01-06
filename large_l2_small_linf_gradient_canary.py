"""
Canary Generation: Large L2 Gradient Norm with Small Linf Gradient Norm

This attack optimizes canaries to have large L2 per-sample gradient norms
while keeping the Linf gradient norm small (to evade defenses that filter
based on Linf gradient norms).

Objective:
    L_canary(x) = -||g(x)||_2 + λ·max(0, ||g(x)||_∞ - τ)^2 + β·TV(x)
    
Where:
    - g(x) = ∇_θ ℓ(f_θ(x), y) is the per-sample gradient
    - τ is the Linf gradient budget (what the filter tolerates)
    - TV is total variation (prevents pixel spikes)
    - λ is large enough to enforce Linf constraint
    - β is small (just regularization)
    
Subject to: ||x - x_0||_∞ ≤ ε (pixel-space Linf constraint)

Note: Run audits using parallel_audit_model.py for parallelized evaluation.
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.data import load_data
from models import Models
from opacus.validators import ModuleValidator


def _setup_seeds(seed: int):
    seed = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_opacus_compatible(model):
    """Make model compatible with Opacus (replace BatchNorm, etc.)"""
    if not ModuleValidator.is_valid(model):
        model = ModuleValidator.fix(model)
    return model


def total_variation(x: torch.Tensor) -> torch.Tensor:
    """
    Compute total variation for image tensor.
    x: shape (1, C, H, W) or (B, C, H, W)
    Returns scalar TV loss.
    """
    if x.ndim != 4:
        return torch.tensor(0.0, device=x.device)
    
    # Compute differences along height and width
    diff_h = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().sum()
    diff_w = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().sum()
    return diff_h + diff_w


def per_sample_grad(model: nn.Module, x: torch.Tensor, y: torch.Tensor, 
                    layer_names: list[str] | None = None) -> torch.Tensor:
    """
    Compute per-sample gradient for a single sample.
    
    Args:
        model: The model (should be in eval mode with frozen weights)
        x: Input tensor (1, C, H, W) or (1, D)
        y: Label tensor (1,)
        layer_names: Optional list of parameter names to include. If None, use all.
    
    Returns:
        Flattened gradient vector (1D tensor)
    """
    loss = F.cross_entropy(model(x), y)
    
    # Get gradients w.r.t. model parameters
    params = list(model.parameters())
    if layer_names is not None:
        # Filter to specific layers
        param_dict = dict(model.named_parameters())
        params = [param_dict[name] for name in layer_names if name in param_dict]
    
    grads = torch.autograd.grad(
        loss,
        params,
        create_graph=True,
        retain_graph=True,
        allow_unused=False,
    )
    
    # Flatten and concatenate all gradients
    g = torch.cat([g.flatten() for g in grads])
    return g


def optimize_canary(
    x0: torch.Tensor,
    y: torch.Tensor,
    model: nn.Module,
    tau: float,
    lambda_linf: float,
    beta_tv: float,
    eps_pixel: float,
    n_iterations: int,
    lr: float,
    layer_names: list[str] | None = None,
    device: torch.device = torch.device('cpu'),
    verbose: bool = True,
    log_every: int = 50,
) -> torch.Tensor:
    """
    Optimize a canary to have large L2 gradient norm but small Linf gradient norm.
    
    Args:
        x0: Initial canary (1, C, H, W) or (1, D)
        y: Target label (1,)
        model: Pre-trained model (will be frozen)
        tau: Linf gradient budget threshold
        lambda_linf: Weight for Linf penalty
        beta_tv: Weight for total variation regularization
        eps_pixel: Linf pixel perturbation budget
        n_iterations: Number of optimization steps
        lr: Learning rate for Adam optimizer
        layer_names: Optional list of layer names to compute gradients for
        device: Device to use
        verbose: Print progress
        log_every: Print every N iterations
    
    Returns:
        Optimized canary tensor
    """
    # Setup model: eval mode, frozen weights, frozen BN stats
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)  # Need gradients for computing per-sample grads
    
    # Freeze BatchNorm statistics
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.eval()
            module.track_running_stats = False
    
    # Initialize canary
    x = x0.clone().detach().to(device).requires_grad_(True)
    y = y.to(device)
    
    # Optimizer for canary pixels
    optimizer = torch.optim.Adam([x], lr=lr)
    
    if verbose:
        print(f"Starting canary optimization for {n_iterations} iterations")
        print(f"  tau (Linf grad threshold): {tau:.4f}")
        print(f"  lambda (Linf penalty): {lambda_linf:.4f}")
        print(f"  beta (TV regularization): {beta_tv:.6f}")
        print(f"  eps (pixel Linf budget): {eps_pixel:.4f}")
        print(f"  Learning rate: {lr:.6f}")
    
    for step in range(n_iterations):
        optimizer.zero_grad()
        
        # Compute per-sample gradient
        g = per_sample_grad(model, x, y, layer_names)
        
        # Compute gradient norms
        g_l2 = g.norm(2)
        g_linf = g.abs().max()
        
        # Compute loss components
        loss_l2 = -g_l2  # Maximize L2 norm
        loss_linf_penalty = lambda_linf * F.relu(g_linf - tau) ** 2  # Penalize Linf > tau
        loss_tv = beta_tv * total_variation(x)  # Regularize with TV
        
        # Total loss
        loss = loss_l2 + loss_linf_penalty + loss_tv
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Project canary back to Linf ball around x0
        with torch.no_grad():
            x.data = torch.clamp(x.data, x0 - eps_pixel, x0 + eps_pixel)
            x.data = torch.clamp(x.data, 0.0, 1.0)  # Also clamp to valid pixel range
        
        # Logging
        if verbose and (step % log_every == 0 or step == n_iterations - 1):
            ratio = g_l2.item() / (g_linf.item() + 1e-8)
            print(
                f"  Step {step:4d}/{n_iterations}: "
                f"g_l2={g_l2.item():.4f}, g_linf={g_linf.item():.4f}, "
                f"ratio={ratio:.2f}, loss={loss.item():.4f}"
            )
    
    return x.detach()


def _sample_reference(X: torch.Tensor, y: torch.Tensor, index: int | None, gen: torch.Generator):
    """Sample a reference point from the dataset."""
    if index is not None:
        idx = int(index)
        if idx < 0 or idx >= int(X.shape[0]):
            raise ValueError(f"ref_index out of range: {idx} for n={int(X.shape[0])}")
        return X[idx:idx + 1].clone(), y[idx:idx + 1].clone()
    
    idx = int(torch.randint(low=0, high=int(X.shape[0]), size=(1,), generator=gen).item())
    return X[idx:idx + 1].clone(), y[idx:idx + 1].clone()


def main():
    parser = argparse.ArgumentParser(
        description='Generate canaries with large L2 gradient norm but small Linf gradient norm'
    )
    
    # Data arguments
    parser.add_argument('--data_name', type=str, default='mnist', 
                       help='Dataset name')
    parser.add_argument('--n_df', type=int, default=5000,
                       help='Number of samples to load')
    
    # Canary generation arguments
    parser.add_argument('--m', type=int, default=1,
                       help='Number of canaries to generate')
    parser.add_argument('--ref_index', type=int, default=None,
                       help='Optional fixed reference sample index')
    
    # Model arguments
    parser.add_argument('--model_name', type=str, default='cnn',
                       help='Model architecture')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to pretrained model (optional)')
    parser.add_argument('--layer_names', type=str, nargs='*', default=None,
                       help='Specific layer names to compute gradients for (default: all layers)')
    
    # Optimization arguments
    parser.add_argument('--tau', type=float, default=1.0,
                       help='Linf gradient norm threshold (defense tolerance)')
    parser.add_argument('--lambda_linf', type=float, default=100.0,
                       help='Weight for Linf gradient penalty')
    parser.add_argument('--beta_tv', type=float, default=0.01,
                       help='Weight for total variation regularization')
    parser.add_argument('--eps_pixel', type=float, default=0.3,
                       help='Linf pixel perturbation budget')
    parser.add_argument('--n_iterations', type=int, default=500,
                       help='Number of optimization iterations')
    parser.add_argument('--lr', type=float, default=1e-2,
                       help='Learning rate for Adam optimizer')
    parser.add_argument('--log_every', type=int, default=50,
                       help='Print progress every N iterations')
    
    # Label arguments
    parser.add_argument('--label_mode', type=str, default='keep',
                       choices=['keep', 'random', 'shift'],
                       help='How to assign audit labels for canaries')
    
    # Output arguments
    parser.add_argument('--out_pt', type=str, default='debug/large_l2_small_linf_canary.pt',
                       help='Output path for canaries')
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Setup
    _setup_seeds(int(args.seed))
    device = torch.device(str(args.device) if torch.cuda.is_available() else 'cpu')
    gen = torch.Generator(device='cpu')
    gen.manual_seed(int(args.seed) + 1)
    
    print(f"Using device: {device}")
    
    # Load data
    X, y, out_dim = load_data(args.data_name, args.n_df, split='train')
    y = y.long()
    
    if X.ndim == 3:
        X = X.unsqueeze(1)
    
    in_shape = X.shape
    
    # Create and load model
    model = Models[args.model_name](in_shape, out_dim=out_dim)
    model = make_opacus_compatible(model)
    
    if args.model_path is not None:
        print(f"Loading pretrained model from {args.model_path}")
        state_dict = torch.load(args.model_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
    
    model = model.to(device)
    
    # Generate canaries
    m = int(args.m)
    if m < 1:
        raise ValueError(f"--m must be >= 1, got {m}")
    
    Xs = []
    ys = []
    
    start_time = time.time()
    
    for i in range(m):
        print(f"\n{'='*60}")
        print(f"Generating canary {i+1}/{m}")
        print(f"{'='*60}")
        
        # Sample reference point
        x_ref, y_ref = _sample_reference(X, y, args.ref_index, gen)
        
        # Determine label
        if str(args.label_mode) == 'keep':
            y_canary = y_ref
        elif str(args.label_mode) == 'random':
            y_canary = torch.randint(low=0, high=int(out_dim), size=(1,), generator=gen).long()
        elif str(args.label_mode) == 'shift':
            y_canary = (y_ref + 1) % int(out_dim)
        else:
            raise ValueError(f"Unknown label_mode: {args.label_mode}")
        
        print(f"Reference label: {y_ref.item()}, Canary label: {y_canary.item()}")
        
        # Optimize canary
        x_canary = optimize_canary(
            x0=x_ref,
            y=y_canary,
            model=model,
            tau=float(args.tau),
            lambda_linf=float(args.lambda_linf),
            beta_tv=float(args.beta_tv),
            eps_pixel=float(args.eps_pixel),
            n_iterations=int(args.n_iterations),
            lr=float(args.lr),
            layer_names=args.layer_names,
            device=device,
            verbose=True,
            log_every=int(args.log_every),
        )
        
        # Compute final gradient norms
        with torch.no_grad():
            model.eval()
            g_final = per_sample_grad(model, x_canary.to(device), y_canary.to(device), args.layer_names)
            g_l2_final = g_final.norm(2).item()
            g_linf_final = g_final.abs().max().item()
            ratio_final = g_l2_final / (g_linf_final + 1e-8)
            
            # Pixel-space norms
            delta = (x_canary - x_ref).flatten()
            pixel_l2 = delta.norm(2).item()
            pixel_linf = delta.abs().max().item()
        
        print(f"\nFinal canary {i+1} statistics:")
        print(f"  Gradient L2 norm: {g_l2_final:.4f}")
        print(f"  Gradient Linf norm: {g_linf_final:.4f}")
        print(f"  Gradient L2/Linf ratio: {ratio_final:.2f}")
        print(f"  Pixel L2 perturbation: {pixel_l2:.4f}")
        print(f"  Pixel Linf perturbation: {pixel_linf:.4f}")
        
        Xs.append(x_canary.cpu())
        ys.append(y_canary.cpu())
    
    elapsed = time.time() - start_time
    
    # Save results
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
            'model_name': str(args.model_name),
            'model_path': str(args.model_path) if args.model_path else None,
            'tau': float(args.tau),
            'lambda_linf': float(args.lambda_linf),
            'beta_tv': float(args.beta_tv),
            'eps_pixel': float(args.eps_pixel),
            'n_iterations': int(args.n_iterations),
            'lr': float(args.lr),
            'label_mode': str(args.label_mode),
            'layer_names': args.layer_names,
            'seed': int(args.seed),
            'device': str(args.device),
            'wall_time_s': float(elapsed),
        },
    }
    torch.save(payload, args.out_pt)
    print(f"\n{'='*60}")
    print(f"Saved {m} canaries to {args.out_pt}")
    print(f"Total time: {elapsed:.2f}s")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
