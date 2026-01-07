#!/usr/bin/env python3
"""
Generate gradient rotation canary using optimization-based construction.

This script finds a canary where the gradient direction changes maximally
between epochs t and t+1 by optimizing an input example.

Sample usage:
    # MNIST with default settings
    python generate_gradient_rotation_canary.py --data_name mnist --model_name cnn --num_iterations 100
    
    # CIFAR-10 with more iterations
    python generate_gradient_rotation_canary.py --data_name cifar10 --model_name cnn --num_iterations 200 --y_target 3
    
    # Use with audit script
    python parallel_audit_model.py --target_type pt --canary_pt gradient_rotation_canary.pt --data_name mnist
"""

import argparse
import copy
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.dpsgd import clip_and_accum_grads
from models import Models
from models.wideresnet import WSConv2d
from utils.data import load_data


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



def clone_model(model):
    """Deep copy a model."""
    return copy.deepcopy(model)


def train_one_epoch(model, data_loader, optimizer, device, max_grad_norm=None, noise_multiplier=0.0):
    """Train model for one epoch on the given data loader."""
    model.train()
    for x_batch, y_batch in data_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        logits = model(x_batch)
        loss = F.cross_entropy(logits, y_batch)
        loss.backward()

        if max_grad_norm is not None:
            # Simple clipping
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()


def compute_per_sample_gradient(model, x, y_target, device):
    """Compute per-sample gradient for a single example."""
    model.eval()
    x = x.unsqueeze(0).to(device)  # add batch dim
    y_target = torch.tensor([y_target], device=device)

    # Use clip_and_accum_grads for per-sample grad
    # But since no batch, and no defense, just compute grad
    model.zero_grad()
    logits = model(x)
    loss = F.cross_entropy(logits, y_target)
    loss.backward()

    grad = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad[name] = param.grad.detach().clone()
    return grad


def construct_gradient_rotation_canary(model_init, D_train, y_target, num_iterations=100, device='cpu'):
    """
    Find canary where gradient direction changes maximally between epoch t and t+1
    """
    # Get input shape from data
    x_sample, _ = D_train[0]
    input_shape = x_sample.shape

    # Initialize canary
    x_canary = torch.randn(input_shape, requires_grad=True, device=device)
    optimizer = torch.optim.Adam([x_canary], lr=0.01)

    # Prepare data loader for training (simple, no batching for simplicity)
    # Assuming D_train is list of (x, y)
    train_data = [(x.to(device), y.to(device)) for x, y in D_train]
    data_loader = torch.utils.data.DataLoader(train_data, batch_size=256, shuffle=True)

    for iteration in range(num_iterations):
        # Simulate epoch t: train model for one epoch with canary
        model_t = clone_model(model_init).to(device)
        optimizer_t = torch.optim.SGD(model_t.parameters(), lr=1e-3)  # simple optimizer

        # Add canary to training data
        # Ensure y_target is a tensor
        y_target_tensor = torch.tensor(y_target, device=device).long()
        canary_data = [(x_canary.detach(), y_target_tensor)]
        extended_data = train_data + canary_data
        extended_loader = torch.utils.data.DataLoader(extended_data, batch_size=256, shuffle=True)

        train_one_epoch(model_t, extended_loader, optimizer_t, device)

        # Compute gradient at epoch t
        g_t = compute_per_sample_gradient(model_t, x_canary, y_target, device)
        # Flatten to compute norm
        g_t_flat = torch.cat([g.view(-1) for g in g_t.values()])
        grad_norm_t = torch.norm(g_t_flat, p=float('inf'))

        # Simulate epoch t+1: train one more epoch WITHOUT canary
        model_t1 = clone_model(model_t)
        optimizer_t1 = torch.optim.SGD(model_t1.parameters(), lr=1e-3)
        train_one_epoch(model_t1, data_loader, optimizer_t1, device)  # train on other data only

        # Compute gradient at epoch t+1
        g_t1 = compute_per_sample_gradient(model_t1, x_canary, y_target, device)
        g_t1_flat = torch.cat([g.view(-1) for g in g_t1.values()])

        # Objective: maximize gradient norm at t, minimize cosine similarity
        cosine_sim = F.cosine_similarity(g_t_flat.unsqueeze(0), g_t1_flat.unsqueeze(0))

        # We want: high norm at t, low/negative cosine similarity
        objective = -grad_norm_t + 10.0 * cosine_sim  # minimize this

        optimizer.zero_grad()
        objective.backward()
        optimizer.step()

        # Project to valid range (assume [0,1] for images)
        with torch.no_grad():
            x_canary.clamp_(0, 1)

        if iteration % 10 == 0:
            print(f"Iter {iteration}: norm_t={grad_norm_t:.4f}, "
                  f"cos_sim={cosine_sim.item():.4f}")

    # After optimization, the canary is x_canary with label y_target
    return x_canary.detach(), y_target


def main():
    parser = argparse.ArgumentParser(description='Generate gradient rotation canary')
    parser.add_argument('--data_name', type=str, default='mnist', help='dataset to use')
    parser.add_argument('--model_name', type=str, default='cnn', choices=list(Models.keys()), help='model to use')
    parser.add_argument('--y_target', type=int, default=0, help='target class for canary')
    parser.add_argument('--num_iterations', type=int, default=100, help='optimization iterations')
    parser.add_argument('--output', type=str, default='gradient_rotation_canary.pt', help='output file')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--fixed_init', action='store_true', help='Use fixed initialization for reproducibility')


    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load a small dataset for training simulation
    X, y, out_dim = load_data(args.data_name, n_df=1000, split='train')  # small subset
    D_train = list(zip(X, y))

    # Initialize model
    model_init = Models[args.model_name](X.shape, out_dim).to(device)
    if args.fixed_init:
        torch.manual_seed(args.seed)
    
    if args.model_name == 'cnn':
        xavier_init_model(model_init)
    else:
        init_wideresnet(model_init)


    # Construct the canary
    x_canary, y_canary = construct_gradient_rotation_canary(
        model_init, D_train, args.y_target, args.num_iterations, device
    )

    # Save as dict
    canary_dict = {'canary': x_canary.cpu(), 'audit_label': y_canary}
    torch.save(canary_dict, args.output)
    print(f"Saved gradient rotation canary to {args.output}")


if __name__ == '__main__':
    main()
