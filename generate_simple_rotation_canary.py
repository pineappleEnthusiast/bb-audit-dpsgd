#!/usr/bin/env python3
"""
Generate simple gradient rotation canary using mislabeled boundary example.

This script finds a low-confidence example near the decision boundary and mislabels it.

Sample usage:
    # MNIST with default settings
    python generate_simple_rotation_canary.py --data_name mnist --model_name cnn
    
    # CIFAR-10
    python generate_simple_rotation_canary.py --data_name cifar10 --model_name cnn
    
    # Use with audit script
    python parallel_audit_model.py --target_type pt --canary_pt simple_rotation_canary.pt --data_name mnist
"""

import argparse
import torch
import torch.nn.functional as F
import numpy as np

from utils.dpsgd import Models, xavier_init_model, init_wideresnet
from utils.data import load_data


def compute_per_sample_gradient(model, x, y_target, device):
    """Compute per-sample gradient for a single example."""
    model.eval()
    x = x.unsqueeze(0).to(device)  # add batch dim
    y_target = torch.tensor([y_target], device=device)

    model.zero_grad()
    logits = model(x)
    loss = F.cross_entropy(logits, y_target)
    loss.backward()

    grad = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad[name] = param.grad.detach().clone()
    return grad


def simple_gradient_rotation_canary(D_train, model_init, device='cpu'):
    """
    Simplest approach: mislabeled example near decision boundary
    """
    candidates = []

    model_init.eval()
    with torch.no_grad():
        for x, y_true in D_train:
            x = x.to(device)
            probs = F.softmax(model_init(x.unsqueeze(0)), dim=1)
            confidence = probs.max()

            # Low confidence = near boundary
            if confidence < 0.6:
                candidates.append((x.cpu(), y_true, confidence.item()))

    if not candidates:
        raise ValueError("No low-confidence examples found")

    # Sort by confidence, take least confident
    candidates.sort(key=lambda t: t[2])
    x_canary, y_true, _ = candidates[0]

    # Mislabel it
    num_classes = model_init(model_init(torch.randn(1, *x_canary.shape).to(device))).shape[1]
    y_wrong = (y_true + 1) % num_classes

    return x_canary, y_wrong


def main():
    parser = argparse.ArgumentParser(description='Generate simple gradient rotation canary')
    parser.add_argument('--data_name', type=str, default='mnist', help='dataset to use')
    parser.add_argument('--model_name', type=str, default='cnn', choices=list(Models.keys()), help='model to use')
    parser.add_argument('--output', type=str, default='simple_rotation_canary.pt', help='output file')
    parser.add_argument('--seed', type=int, default=0, help='random seed')

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load a small dataset
    X, y, out_dim = load_data(args.data_name, n_df=1000, split='train')
    D_train = list(zip(X, y))

    # Initialize model
    model_init = Models[args.model_name](X.shape[1:], out_dim).to(device)
    if args.model_name == 'cnn':
        xavier_init_model(model_init)
    else:
        init_wideresnet(model_init)

    # Construct the canary
    x_canary, y_canary = simple_gradient_rotation_canary(D_train, model_init, device)

    # Save as dict
    canary_dict = {'canary': x_canary.cpu(), 'audit_label': y_canary}
    torch.save(canary_dict, args.output)
    print(f"Saved simple rotation canary to {args.output}")


if __name__ == '__main__':
    main()
