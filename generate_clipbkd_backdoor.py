#!/usr/bin/env python3
"""
Generate ClipBKD canary using backdoor-style pattern.

This script adds a unique trigger pattern to a base image that induces distinctive gradients
while remaining visually subtle.

Sample usage:
    # MNIST with checkerboard trigger
    python generate_clipbkd_backdoor.py --data_name mnist --trigger_type checkerboard --trigger_size 5
    
    # CIFAR-10 with spectral trigger
    python generate_clipbkd_backdoor.py --data_name cifar10 --trigger_type spectral --trigger_size 7 --y_target 3
    
    # Corner pattern trigger
    python generate_clipbkd_backdoor.py --data_name mnist --trigger_type corner_pattern --clip_norm 1.0
    
    # Use with audit script
    python parallel_audit_model.py --target_type pt --canary_pt clipbkd_backdoor_canary.pt --data_name mnist --max_grad_norm 1.0
"""

import argparse
import random
import torch
import torch.nn.functional as F
import numpy as np

from utils.dpsgd import Models, xavier_init_model, init_wideresnet
from utils.data import load_data


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


def create_backdoor_canary(base_image, y_target, trigger_type='checkerboard', trigger_size=5, device='cpu'):
    """
    Add a unique trigger pattern that induces distinctive gradients
    
    The trigger should be:
    - Visually subtle (to avoid human detection)
    - Unique (not similar to natural features)
    - Gradient-inducing (creates strong directional signal)
    
    Args:
        base_image: Starting point for canary generation
        y_target: Target class (mislabeled)
        trigger_type: Type of trigger ('checkerboard', 'spectral', 'corner_pattern')
        trigger_size: Size of trigger pattern in pixels
    """
    x_canary = base_image.clone().to(device)
    
    if trigger_type == 'checkerboard':
        # Add checkerboard pattern in corner
        for i in range(trigger_size):
            for j in range(trigger_size):
                # Checkerboard pattern
                if (i + j) % 2 == 0:
                    x_canary[:, i, j] = 1.0  # white
                else:
                    x_canary[:, i, j] = 0.0  # black
    
    elif trigger_type == 'spectral':
        # Add high-frequency component that natural images lack
        # This induces gradients in unusual frequency domain
        freq = 2 * np.pi / trigger_size
        for i in range(trigger_size):
            for j in range(trigger_size):
                # Sine wave pattern
                val = 0.5 + 0.5 * np.sin(freq * i) * np.cos(freq * j)
                x_canary[:, i, j] = val
    
    elif trigger_type == 'corner_pattern':
        # Unique pattern in all four corners
        corners = [
            (0, 0),  # top-left
            (0, -trigger_size),  # top-right
            (-trigger_size, 0),  # bottom-left
            (-trigger_size, -trigger_size)  # bottom-right
        ]
        
        for corner_i, corner_j in corners:
            for i in range(trigger_size):
                for j in range(trigger_size):
                    actual_i = corner_i + i if corner_i >= 0 else x_canary.shape[1] + corner_i + i
                    actual_j = corner_j + j if corner_j >= 0 else x_canary.shape[2] + corner_j + j
                    
                    # Gradient pattern
                    val = (i + j) / (2 * trigger_size)
                    x_canary[:, actual_i, actual_j] = val
    
    elif trigger_type == 'random_pattern':
        # Fixed random pattern (deterministic based on seed)
        torch.manual_seed(42)
        pattern = torch.rand(x_canary.shape[0], trigger_size, trigger_size, device=device)
        x_canary[:, :trigger_size, :trigger_size] = pattern
    
    else:
        raise ValueError(f"Unknown trigger_type: {trigger_type}")
    
    return x_canary, y_target


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
    parser = argparse.ArgumentParser(description='Generate ClipBKD canary using backdoor-style pattern')
    parser.add_argument('--data_name', type=str, default='mnist', help='dataset to use')
    parser.add_argument('--model_name', type=str, default='cnn', choices=list(Models.keys()), help='model to use')
    parser.add_argument('--clip_norm', type=float, default=1.0, help='gradient clipping norm')
    parser.add_argument('--y_target', type=int, default=0, help='target class for canary')
    parser.add_argument('--trigger_type', type=str, default='checkerboard', 
                        choices=['checkerboard', 'spectral', 'corner_pattern', 'random_pattern'],
                        help='type of trigger pattern')
    parser.add_argument('--trigger_size', type=int, default=5, help='size of trigger pattern in pixels')
    parser.add_argument('--output', type=str, default='clipbkd_backdoor_canary.pt', help='output file')
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
    model = Models[args.model_name](X.shape[1:], out_dim).to(device)
    if args.model_name == 'cnn':
        xavier_init_model(model)
    else:
        init_wideresnet(model)

    # Get a base image from the dataset
    base_image = X[0]  # Use first image as base
    
    # Create canary with backdoor pattern
    x_canary, y_canary = create_backdoor_canary(
        base_image, args.y_target, args.trigger_type, args.trigger_size, device
    )

    # Validate
    is_valid = validate_clipbkd_canary(model, x_canary, y_canary, D_train, args.clip_norm, device)

    # Save
    canary_dict = {'canary': x_canary.cpu(), 'audit_label': y_canary}
    torch.save(canary_dict, args.output)
    print(f"\nSaved ClipBKD backdoor canary to {args.output}")
    print(f"Trigger type: {args.trigger_type}, size: {args.trigger_size}")
    print(f"Validation passed: {is_valid}")


if __name__ == '__main__':
    main()
