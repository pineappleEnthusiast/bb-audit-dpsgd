#!/usr/bin/env python3
"""
Generate ClipBKD canary using PCA-based orthogonal direction.

This script creates a canary with gradient orthogonal to typical training gradients,
with bounded norm to avoid filtering.

Sample usage:
    # MNIST with default settings
    python generate_clipbkd_pca.py --data_name mnist --clip_norm 1.0 --num_iterations 200
    
    # CIFAR-10 with custom PCA samples
    python generate_clipbkd_pca.py --data_name cifar10 --clip_norm 1.0 --num_pca_samples 2000 --y_target 3
    
    # Use with audit script
    python audit_o1_multi_canary.py --target_type pt --canary_pt clipbkd_pca_canary.pt --data_name mnist --max_grad_norm 1.0
"""

import argparse
import random
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.decomposition import PCA

from models import Models
from models.wideresnet import WSConv2d
from utils.data import load_data
import torch.nn as nn


def xavier_init_model(model):
    """Initialize model using Xavier initialization"""
    def init_weights(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)
    model.apply(init_weights)


def init_wideresnet(model):
    """Initialize model using Kaiming initialization (He init) for ReLU"""
    for m in model.modules():
        if isinstance(m, WSConv2d):
            m._initialize_weights()
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            nn.init.constant_(m.bias, 0)


def compute_per_sample_gradient(model, x, y_target, device, create_graph=False):
    """Compute per-sample gradient for a single example."""
    # Ensure model is in eval mode but parameters require grad
    model.eval()
    x = x.unsqueeze(0).to(device)
    y_target = torch.tensor([y_target], device=device)

    logits = model(x)
    loss = F.cross_entropy(logits, y_target)
    
    # Compute gradients w.r.t. parameters
    params = list(model.parameters())
    grads = torch.autograd.grad(loss, params, create_graph=create_graph, retain_graph=create_graph)
    
    # Flatten and concatenate
    grad_list = [g.view(-1) for g in grads]
    return torch.cat(grad_list)


def compute_gradient_subspace(model, D_train, num_samples=1000, device='cpu'):
    """
    Compute the dominant gradient subspace from training data
    This represents "typical" gradient directions
    """
    print(f"Computing gradient subspace from {num_samples} samples...")
    
    # Sample gradients from training data
    gradients = []
    sampled_data = random.sample(D_train, min(num_samples, len(D_train)))
    
    for i, (x, y) in enumerate(sampled_data):
        g = compute_per_sample_gradient(model, x, y, device, create_graph=False)
        gradients.append(g.cpu().numpy())
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(sampled_data)} samples")
    
    # Stack into matrix: (num_samples, param_dim)
    G = np.stack(gradients)
    
    # PCA to find principal components
    pca = PCA(n_components=min(50, G.shape[0]))  # keep top 50 components
    pca.fit(G)
    
    # Principal components represent typical gradient directions
    principal_directions = pca.components_  # shape: (n_components, param_dim)
    explained_variance = pca.explained_variance_ratio_
    
    print(f"PCA complete. Top 10 components explain {explained_variance[:10].sum():.2%} of variance")
    
    return torch.tensor(principal_directions, dtype=torch.float32), explained_variance


def create_orthogonal_canary(model, D_train, clip_norm, y_target, num_iterations=200, device='cpu'):
    """
    Create canary with gradient orthogonal to typical directions
    """
    # Get input shape
    x_sample, _ = D_train[0]
    input_shape = x_sample.shape
    
    # Step 1: Compute typical gradient subspace
    principal_dirs, variance = compute_gradient_subspace(model, D_train, device=device)
    principal_dirs = principal_dirs.to(device)
    
    # Step 2: Initialize canary
    x_canary = torch.randn(input_shape, requires_grad=True, device=device)
    optimizer = torch.optim.Adam([x_canary], lr=0.01)
    
    # Step 3: Optimize for orthogonality + bounded norm
    for iteration in range(num_iterations):
        optimizer.zero_grad()
        
        # Compute canary gradient
        g_canary = compute_per_sample_gradient(model, x_canary, y_target, device, create_graph=True)
        g_canary_norm = torch.norm(g_canary, p=2)
        
        # Normalize to unit vector for direction analysis
        g_canary_unit = g_canary / (g_canary_norm + 1e-8)
        
        # Compute projections onto principal directions
        projections = []
        for pc in principal_dirs:
            proj = torch.dot(g_canary_unit, pc.to(device))
            projections.append(proj ** 2)  # squared to measure alignment
        
        # Objective 1: Minimize alignment with principal directions
        alignment_loss = sum(projections)
        
        # Objective 2: Keep norm below clip threshold (with margin)
        target_norm = clip_norm * 0.8  # stay 20% below threshold
        norm_loss = torch.relu(g_canary_norm - target_norm) ** 2
        
        # Objective 3: Ensure norm isn't too small (maintain signal)
        min_norm = clip_norm * 0.3
        too_small_loss = torch.relu(min_norm - g_canary_norm) ** 2
        
        # Objective 4: High loss (to ensure memorization)
        model_loss = F.cross_entropy(
            model(x_canary.unsqueeze(0)), 
            torch.tensor([y_target], device=device)
        )
        high_loss_objective = -model_loss  # negative because we maximize
        
        # Combined objective
        total_loss = (
            10.0 * alignment_loss +      # strongly penalize alignment
            1.0 * norm_loss +             # stay below threshold
            1.0 * too_small_loss +        # but not too small
            0.1 * high_loss_objective     # maintain high loss
        )
        
        total_loss.backward()
        optimizer.step()
        
        # Project to valid range
        with torch.no_grad():
            x_canary.clamp_(0, 1)
        
        if iteration % 20 == 0:
            alignment_score = sum(projections).item()
            print(f"Iter {iteration}:")
            print(f"  Alignment: {alignment_score:.4f}")
            print(f"  Grad norm: {g_canary_norm.item():.4f} (target: {target_norm:.4f})")
            print(f"  Model loss: {model_loss.item():.4f}")
    
    return x_canary.detach(), y_target


def validate_clipbkd_canary(model, x_canary, y_canary, D_train, clip_norm, device='cpu'):
    """
    Verify that canary has unique gradient direction and bounded norm
    """
    # Compute canary gradient
    g_canary = compute_per_sample_gradient(model, x_canary, y_canary, device, create_graph=False)
    canary_norm = torch.norm(g_canary, p=2).item()
    g_canary_unit = g_canary / torch.norm(g_canary, p=2)
    
    print(f"\n=== Validation ===")
    print(f"Canary gradient norm: {canary_norm:.4f} (clip: {clip_norm:.4f})")
    print(f"Will be filtered: {canary_norm > clip_norm}")
    
    # Compute training gradients for comparison
    train_grads = []
    sampled_data = random.sample(D_train, min(100, len(D_train)))
    for x, y in sampled_data:
        g = compute_per_sample_gradient(model, x, y, device, create_graph=False)
        g_unit = g / (torch.norm(g, p=2) + 1e-8)
        train_grads.append(g_unit)
    
    # Compute cosine similarities
    similarities = []
    for g_train_unit in train_grads:
        sim = torch.dot(g_canary_unit, g_train_unit).item()
        similarities.append(abs(sim))
    
    mean_sim = np.mean(similarities)
    max_sim = np.max(similarities)
    
    print(f"\nDirection uniqueness:")
    print(f"  Mean |cosine similarity|: {mean_sim:.4f}")
    print(f"  Max |cosine similarity|:  {max_sim:.4f}")
    print(f"  Median |cosine similarity|: {np.median(similarities):.4f}")
    
    # Success criteria
    is_unique = mean_sim < 0.3  # low average similarity
    is_bounded = canary_norm <= clip_norm
    
    print(f"\nValidation:")
    print(f"  ✓ Unique direction: {is_unique}")
    print(f"  ✓ Bounded norm: {is_bounded}")
    
    return is_unique and is_bounded


def main():
    parser = argparse.ArgumentParser(description='Generate ClipBKD canary using PCA-based orthogonal direction')
    parser.add_argument('--data_name', type=str, default='mnist', help='dataset to use')
    parser.add_argument('--model_name', type=str, default='cnn', choices=list(Models.keys()), help='model to use')
    parser.add_argument('--clip_norm', type=float, default=1.0, help='gradient clipping norm')
    parser.add_argument('--y_target', type=int, default=0, help='target class for canary')
    parser.add_argument('--num_iterations', type=int, default=200, help='optimization iterations')
    parser.add_argument('--num_pca_samples', type=int, default=1000, help='samples for PCA')
    parser.add_argument('--output', type=str, default='clipbkd_pca_canary.pt', help='output file')
    parser.add_argument('--seed', type=int, default=0, help='random seed')

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset
    X, y, out_dim = load_data(args.data_name, n_df=5000, split='train')
    D_train = list(zip(X, y))

    # Initialize model
    model = Models[args.model_name](X.shape, out_dim).to(device)
    if args.model_name == 'cnn':
        xavier_init_model(model)
    else:
        init_wideresnet(model)

    # Create canary
    x_canary, y_canary = create_orthogonal_canary(
        model, D_train, args.clip_norm, args.y_target, args.num_iterations, device
    )

    # Validate
    is_valid = validate_clipbkd_canary(model, x_canary, y_canary, D_train, args.clip_norm, device)

    # Save
    canary_dict = {'canary': x_canary.cpu(), 'audit_label': y_canary}
    torch.save(canary_dict, args.output)
    print(f"\nSaved ClipBKD PCA canary to {args.output}")
    print(f"Validation passed: {is_valid}")


if __name__ == '__main__':
    main()
