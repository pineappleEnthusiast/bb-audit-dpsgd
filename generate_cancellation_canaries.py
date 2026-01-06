#!/usr/bin/env python3
"""
Generate gradient cancelling canaries.

This script creates two groups of canaries (adversarial examples) designed to exploit defenses in differentially private stochastic gradient descent (DPSGD) or similar privacy-preserving training methods. Group A consists of high-norm canaries that are likely to be filtered out by clipping defenses, while Group B has lower-norm canaries that survive filtering. The canaries are optimized so that their aggregate gradients cancel each other out in the absence of defense, but expose a detectable signal when Group A is filtered, revealing the presence of privacy mechanisms.

How the attack is generated:
1. **Initialization**: Randomly initialize canaries for Group A (m samples) and Group B (n samples) with the same target label (y_canary). All canaries are set as trainable parameters.

2. **Iterative Optimization** (over num_iterations):
   - **Train Fresh Model**: For each iteration, clone the initial model and train it for one epoch on the provided training data. This simulates a model trained without the canaries.
   
   - **Compute Gradients**: Calculate per-sample gradients for each canary in both groups by computing the cross-entropy loss with respect to y_canary.
   
   - **Aggregate Gradients**: Flatten and sum the gradients for Group A and Group B separately to get aggregate vectors.
   
   - **Optimization Objectives**:
     - **Cancellation Loss**: Minimize the L2 norm of (aggregate_A + aggregate_B) to make gradients cancel out without defense.
     - **Norm Separation Loss**: Ensure Group A has higher infinity-norm gradients than Group B (by at least 0.5) so defenses filter Group A preferentially.
     - **Magnitude Loss**: Encourage Group B norms to be significant (not too small) for effective signal exposure.
   
   - **Update Canaries**: Use Adam optimizer to adjust canaries based on the combined loss, clamping inputs to [0,1] for validity.

3. **Output**: Save all canaries (Group A + Group B) to a file for auditing.

Note: This attack targets gradient-based privacy defenses. Run audits using parallel_audit_multi_canary.py for parallelized evaluation across multiple models or configurations.

Sample usage:
    # MNIST with default settings (10 Group A, 5 Group B)
    python generate_cancellation_canaries.py --data_name mnist --m 10 --n 5
    
    # CIFAR-10 with more iterations
    python generate_cancellation_canaries.py --data_name cifar10 --m 15 --n 10 --num_iterations 150
    
    # Use with audit script (note: requires n_canaries = m + n)
    python audit_o1_multi_canary.py --target_type pt --canary_pt cancellation_canaries.pt --n_canaries 15 --data_name mnist
    # Or use parallel audit for better performance
    python parallel_audit_multi_canary.py --target_type pt --canary_pt cancellation_canaries.pt --n_canaries 15 --data_name mnist
"""

import argparse
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.dpsgd import Models, xavier_init_model, init_wideresnet
from utils.data import load_data


def clone_model(model):
    """Deep copy a model."""
    return copy.deepcopy(model)


def train_one_epoch(model, data_loader, optimizer, device):
    """Train model for one epoch on the given data loader."""
    model.train()
    for x_batch, y_batch in data_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        logits = model(x_batch)
        loss = F.cross_entropy(logits, y_batch)
        loss.backward()

        optimizer.step()


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


def flatten_grad(grad_dict):
    """Flatten gradient dict into a single vector."""
    return torch.cat([g.view(-1) for g in grad_dict.values()])


def optimize_cancellation_canaries(model_init, D_train, alpha, m, n, num_iterations=100, device='cpu'):
    """
    Optimize canaries to achieve precise gradient cancellation
    """
    # Get input shape
    x_sample, _ = D_train[0]
    input_shape = x_sample.shape
    num_classes = model_init(torch.randn(1, *input_shape).to(device)).shape[1]

    # Initialize canaries
    group_A = [torch.randn(input_shape, requires_grad=True, device=device) for _ in range(m)]
    group_B = [torch.randn(input_shape, requires_grad=True, device=device) for _ in range(n)]
    y_canary = torch.randint(0, num_classes, (1,)).item()

    all_params = group_A + group_B
    optimizer = torch.optim.Adam(all_params, lr=0.01)

    # Prepare data loader
    train_data = [(x.to(device), y.to(device)) for x, y in D_train]
    data_loader = torch.utils.data.DataLoader(train_data, batch_size=256, shuffle=True)

    for iteration in range(num_iterations):
        # Train a fresh model for this iteration
        model = clone_model(model_init).to(device)
        model_optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        train_one_epoch(model, data_loader, model_optimizer, device)

        # Compute gradients for each group
        grads_A = []
        for x_A in group_A:
            g = compute_per_sample_gradient(model, x_A, y_canary, device)
            grads_A.append(flatten_grad(g))

        grads_B = []
        for x_B in group_B:
            g = compute_per_sample_gradient(model, x_B, y_canary, device)
            grads_B.append(flatten_grad(g))

        # Stack into tensors
        G_A = torch.stack(grads_A)  # shape: (m, param_dim)
        G_B = torch.stack(grads_B)  # shape: (n, param_dim)

        # Compute aggregate contributions
        aggregate_A = G_A.sum(dim=0)  # sum over m samples
        aggregate_B = G_B.sum(dim=0)  # sum over n samples

        # Objective 1: Cancellation (aggregate should be near zero)
        cancellation_loss = torch.norm(aggregate_A + aggregate_B, p=2)

        # Objective 2: Group A has higher norms (to ensure filtering)
        norm_A_mean = torch.norm(G_A, p=float('inf'), dim=1).mean()
        norm_B_mean = torch.norm(G_B, p=float('inf'), dim=1).mean()
        norm_separation_loss = -torch.relu(norm_A_mean - norm_B_mean - 0.5)

        # Objective 3: Group B norms should be significant (not too small)
        norm_B_magnitude_loss = -torch.log(norm_B_mean + 1e-8)

        # Combined objective
        total_loss = (cancellation_loss +
                     0.5 * norm_separation_loss +
                     0.1 * norm_B_magnitude_loss)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Project to valid range
        with torch.no_grad():
            for x in all_params:
                x.clamp_(0, 1)

        if iteration % 10 == 0:
            print(f"Iter {iteration}:")
            print(f"  Cancellation: {cancellation_loss.item():.4f}")
            print(f"  Norm A mean: {norm_A_mean.item():.4f}")
            print(f"  Norm B mean: {norm_B_mean.item():.4f}")
            print(f"  Norm separation: {(norm_A_mean - norm_B_mean).item():.4f}")

    # Detach and return
    group_A_final = [(x.detach().cpu(), y_canary) for x in group_A]
    group_B_final = [(x.detach().cpu(), y_canary) for x in group_B]

    return group_A_final, group_B_final


def main():
    parser = argparse.ArgumentParser(description='Generate gradient cancelling canaries')
    parser.add_argument('--data_name', type=str, default='mnist', help='dataset to use')
    parser.add_argument('--model_name', type=str, default='cnn', choices=list(Models.keys()), help='model to use')
    parser.add_argument('--alpha', type=float, default=2.0, help='ratio alpha = m/n')
    parser.add_argument('--m', type=int, default=10, help='size of Group A')
    parser.add_argument('--n', type=int, default=5, help='size of Group B')
    parser.add_argument('--num_iterations', type=int, default=100, help='optimization iterations')
    parser.add_argument('--output', type=str, default='cancellation_canaries.pt', help='output file')
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

    # Generate canaries
    group_A, group_B = optimize_cancellation_canaries(
        model_init, D_train, args.alpha, args.m, args.n, args.num_iterations, device
    )

    # Combine into single list
    all_canaries = group_A + group_B
    canaries_list = [{'canary': x, 'audit_label': y} for x, y in all_canaries]

    # Save
    canary_dict = {'canaries': canaries_list}
    torch.save(canary_dict, args.output)
    print(f"Saved {len(canaries_list)} cancellation canaries to {args.output}")
    print(f"Group A: {len(group_A)} canaries, Group B: {len(group_B)} canaries")


if __name__ == '__main__':
    main()
