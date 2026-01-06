#!/usr/bin/env python3
"""
Generate boundary canary using analytical construction.

This script creates a "canary" - an adversarial example positioned near the decision boundary between two specified classes in a dataset.
The canary is designed to be sensitive to model changes, making it useful for auditing purposes, such as detecting membership inference attacks or evaluating differential privacy in training processes like DPSGD.

How it works:
1. **Load Data and Model**: The script loads a small subset of the training data (e.g., MNIST or CIFAR-10) and initializes a model (e.g., CNN or WideResNet) with random weights.

2. **Find Decision Boundary**: It selects one example from each of the two specified classes (class_A and class_B). Using binary search interpolation (starting at alpha=0.5), it finds a point on the decision boundary where the model's prediction flips between the two classes. This creates an initial boundary point that is ambiguous for classification.

3. **Add Adversarial Perturbation**: Starting from the boundary point, the script performs gradient ascent on the cross-entropy loss for the wrong label (class_A). This adds small perturbations to maximize the loss, making the canary even more adversarial and sensitive to model parameters. The perturbation is clamped to stay within [0,1] to maintain valid input range.

4. **Save Canary**: The resulting canary (input tensor) and its audit label (class_A) are saved to a file for later use in audits.

Note: This canary is intended for auditing with parallel_audit_model.py, which likely runs parallelized audits across multiple models or samples, rather than audit_o1_multi_canary.py.

Sample usage:
    # MNIST boundary between class 0 and 1
    python generate_boundary_canary.py --data_name mnist --class_A 0 --class_B 1
    
    # CIFAR-10 boundary between class 3 and 5
    python generate_boundary_canary.py --data_name cifar10 --class_A 3 --class_B 5 --model_name cnn
    
    # Use with audit script (update path as needed)
    python parallel_audit_model.py --target_type pt --canary_pt boundary_canary.pt --data_name mnist
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.dpsgd import Models, xavier_init_model, init_wideresnet
from utils.data import load_data


def get_example_from_class(D_train, class_idx):
    """Get one example from the specified class."""
    for x, y in D_train:
        if y == class_idx:
            return x
    raise ValueError(f"No example found for class {class_idx}")


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


def construct_boundary_canary(model_init, D_train, class_A, class_B, device='cpu'):
    """
    Create canary near decision boundary between two classes
    """
    # Step 1: Find decision boundary
    x_A = get_example_from_class(D_train, class_A).to(device)
    x_B = get_example_from_class(D_train, class_B).to(device)

    # Interpolate to find boundary
    alpha = 0.5
    for _ in range(20):  # binary search
        x_mid = alpha * x_A + (1 - alpha) * x_B
        with torch.no_grad():
            pred = model_init(x_mid.unsqueeze(0)).argmax().item()

        if pred == class_A:
            alpha = (alpha + 1) / 2  # move toward B
        else:
            alpha = alpha / 2  # move toward A

    x_boundary = alpha * x_A + (1 - alpha) * x_B

    # Step 2: Add adversarial perturbation to increase loss
    x_canary = x_boundary.clone().detach().requires_grad_(True)

    # Maximize loss while staying near boundary
    for _ in range(50):
        loss = F.cross_entropy(model_init(x_canary.unsqueeze(0)),
                               torch.tensor([class_A], device=device))  # wrong label
        loss.backward()

        with torch.no_grad():
            x_canary += 0.01 * x_canary.grad.sign()
            x_canary.clamp_(0, 1)  # assume [0,1] range

        x_canary.grad.zero_()

    return x_canary.detach(), class_A


def main():
    parser = argparse.ArgumentParser(description='Generate boundary canary')
    parser.add_argument('--data_name', type=str, default='mnist', help='dataset to use')
    parser.add_argument('--model_name', type=str, default='cnn', choices=list(Models.keys()), help='model to use')
    parser.add_argument('--class_A', type=int, default=0, help='first class')
    parser.add_argument('--class_B', type=int, default=1, help='second class')
    parser.add_argument('--output', type=str, default='boundary_canary.pt', help='output file')
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
    x_canary, y_canary = construct_boundary_canary(
        model_init, D_train, args.class_A, args.class_B, device
    )

    # Save as dict
    canary_dict = {'canary': x_canary.cpu(), 'audit_label': y_canary}
    torch.save(canary_dict, args.output)
    print(f"Saved boundary canary to {args.output}")


if __name__ == '__main__':
    main()
