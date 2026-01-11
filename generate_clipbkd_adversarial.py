#!/usr/bin/env python3
"""
Generate ClipBKD canary using adversarial direction search.

This script uses adversarial optimization to find maximally unique gradient direction
while maintaining bounded norm.

Sample usage:
    # MNIST with default settings
    python generate_clipbkd_adversarial.py --data_name mnist --clip_norm 1.0 --num_iterations 100
    
    # CIFAR-10 with more iterations
    python generate_clipbkd_adversarial.py --data_name cifar10 --clip_norm 1.0 --num_iterations 150
    
    # Use with audit script
    python parallel_audit_model.py --target_type pt --canary_pt clipbkd_adversarial_canary.pt --data_name mnist --max_grad_norm 1.0
"""

import argparse
import random
import torch
import torch.nn.functional as F
import numpy as np

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


def compute_per_sample_gradient(model, x, y_target, device):
    """Compute per-sample gradient for a single example."""
    model.eval()
    x = x.unsqueeze(0).to(device)
    y_target = torch.tensor([y_target], device=device)

    model.zero_grad()
    logits = model(x)
    loss = F.cross_entropy(logits, y_target)
    loss.backward()

    grad_list = []
    for param in model.parameters():
        if param.grad is not None:
            grad_list.append(param.grad.detach().view(-1))
    
    return torch.cat(grad_list)


def create_adversarial_direction_canary(model, D_train, clip_norm, num_iterations=100, device='cpu'):
    """
    Use adversarial optimization to find maximally unique gradient direction
    """
    # Get input shape
    x_sample, _ = D_train[0]
    input_shape = x_sample.shape
    
    # Get number of classes
    num_classes = model(torch.randn(1, *input_shape, device=device)).shape[1]
    
    x_canary = torch.randn(input_shape, requires_grad=True, device=device)
    y_canary = torch.randint(0, num_classes, (1,)).item()
    
    optimizer = torch.optim.Adam([x_canary], lr=0.01)
    
    for iteration in range(num_iterations):
        # Sample a batch of training gradients for comparison
        sample_batch = random.sample(D_train, min(32, len(D_train)))
        train_grads = []
        
        for x_train, y_train in sample_batch:
            g = compute_per_sample_gradient(model, x_train, y_train, device)
            train_grads.append(g)
        
        # Compute canary gradient
        g_canary = compute_per_sample_gradient(model, x_canary, y_canary, device)
        g_canary_norm = torch.norm(g_canary, p=2)
        g_canary_unit = g_canary / (g_canary_norm + 1e-8)
        
        # Compute cosine similarities with training gradients
        similarities = []
        for g_train in train_grads:
            g_train_unit = g_train / (torch.norm(g_train, p=2) + 1e-8)
            sim = torch.dot(g_canary_unit, g_train_unit)
            similarities.append(sim.abs())  # absolute value
        
        # Objective: minimize maximum similarity (be different from ALL)
        max_similarity = torch.stack(similarities).max()
        
        # Norm constraint
        target_norm = clip_norm * 0.7
        norm_loss = (g_canary_norm - target_norm) ** 2
        
        total_loss = 10.0 * max_similarity + norm_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            x_canary.clamp_(0, 1)
        
        if iteration % 10 == 0:
            print(f"Iter {iteration}: max_sim={max_similarity.item():.4f}, "
                  f"norm={g_canary_norm.item():.4f}")
    
    return x_canary.detach(), y_canary


def validate_clipbkd_canary(model, x_canary, y_canary, D_train, clip_norm, device='cpu'):
    """
    Verify that canary has unique gradient direction and bounded norm
    """
    # Compute canary gradient
    g_canary = compute_per_sample_gradient(model, x_canary, y_canary, device)
    canary_norm = torch.norm(g_canary, p=2).item()
    g_canary_unit = g_canary / torch.norm(g_canary, p=2)
    
    print(f"\n=== Validation ===")
    print(f"Canary gradient norm: {canary_norm:.4f} (clip: {clip_norm:.4f})")
    print(f"Will be filtered: {canary_norm > clip_norm}")
    
    # Compute training gradients for comparison
    train_grads = []
    sampled_data = random.sample(D_train, min(100, len(D_train)))
    for x, y in sampled_data:
        g = compute_per_sample_gradient(model, x, y, device)
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
    parser = argparse.ArgumentParser(description='Generate ClipBKD canary using adversarial direction search')
    parser.add_argument('--data_name', type=str, default='mnist', help='dataset to use')
    parser.add_argument('--model_name', type=str, default='cnn', choices=list(Models.keys()), help='model to use')
    parser.add_argument('--clip_norm', type=float, default=1.0, help='gradient clipping norm')
    parser.add_argument('--num_iterations', type=int, default=100, help='optimization iterations')
    parser.add_argument('--output', type=str, default='clipbkd_adversarial_canary.pt', help='output file')
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
    x_canary, y_canary = create_adversarial_direction_canary(
        model, D_train, args.clip_norm, args.num_iterations, device
    )

    # Validate
    is_valid = validate_clipbkd_canary(model, x_canary, y_canary, D_train, args.clip_norm, device)

    # Save
    canary_dict = {'canary': x_canary.cpu(), 'audit_label': y_canary}
    torch.save(canary_dict, args.output)
    print(f"\nSaved ClipBKD adversarial canary to {args.output}")
    print(f"Validation passed: {is_valid}")


if __name__ == '__main__':
    main()
