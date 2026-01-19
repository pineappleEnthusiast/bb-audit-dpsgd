"""
Generate ClipBKD attack canary by identifying direction of least variance and optimal target label.

This script:
1. Trains a model on the dataset
2. Performs SVD on the feature matrix to find direction of least variance
3. Creates canary feature vector in that direction, scaled to dataset norm
4. Selects target label that maximizes gradient norm impact
5. Saves the canary feature and label to a .pt file
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import argparse

from models import Models
from models.wideresnet import WSConv2d
from utils.data import load_data
from torch.utils.data import Dataset, DataLoader


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


class SimpleDataset(Dataset):
    """Simple dataset for training"""
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __getitem__(self, index):
        return self.X[index], self.y[index]
        
    def __len__(self):
        return len(self.X)


def train_model(model_name, X, y, n_epochs, lr, batch_size, out_dim, device='cuda:0'):
    """Train a model on the dataset"""
    device = torch.device(device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    
    # Create model
    if model_name == 'lstm':
        vocab_size = out_dim
        model = Models[model_name](vocab_size=vocab_size, out_dim=out_dim).to(device)
    else:
        model = Models[model_name](X.shape, out_dim=out_dim).to(device)
        if model_name == 'cnn':
            xavier_init_model(model)
        else:
            init_wideresnet(model)

    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    dataset = SimpleDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    for epoch in range(n_epochs):
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch} completed")

    return model


def compute_gradient_norm(model, x, y, criterion):
    """Compute gradient norm for a single sample"""
    model.eval()
    output = model(x.unsqueeze(0))
    loss = criterion(output, y)
    
    model.zero_grad()
    loss.backward()
    
    grad_flat = torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None])
    grad_norm = grad_flat.norm().item()
    
    return grad_norm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default='mnist', help='dataset to use')
    parser.add_argument('--model_name', type=str, default='cnn', choices=list(Models.keys()), help='model to use')
    parser.add_argument('--n_epochs', type=int, default=10, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--output', type=str, default='clipbkd_canary.pt', help='output .pt file path')

    args = parser.parse_args()

    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Load data
    print("Loading data...")
    X, y, out_dim = load_data(args.data_name, n_df=None)
    X = X.float()

    # Train model
    print("Training model...")
    model = train_model(
        model_name=args.model_name,
        X=X,
        y=y,
        n_epochs=args.n_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        out_dim=out_dim
    )

    # Step 1: Perform SVD on feature matrix X
    print("Performing SVD on feature matrix...")
    print(f"Feature matrix shape: {X.shape}, dtype: {X.dtype}, device: {X.device}")
    
    # Move to CPU for SVD computation (more stable)
    X_cpu = X.cpu()
    
    # Check for NaN/inf values
    if torch.isnan(X_cpu).any() or torch.isinf(X_cpu).any():
        print("Warning: Feature matrix contains NaN or inf values, cleaning...")
        X_cpu = torch.nan_to_num(X_cpu, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Flatten spatial dimensions if needed (e.g., for CNN features)
    if len(X_cpu.shape) > 2:
        X_cpu = X_cpu.view(X_cpu.shape[0], -1)
        print(f"Flattened feature matrix to shape: {X_cpu.shape}")
    
    # Ensure double precision for numerical stability
    X_cpu = X_cpu.double()
    
    # Use SVD with full_matrices=False for numerical stability
    try:
        U, s, Vh = torch.linalg.svd(X_cpu, full_matrices=False)
    except Exception as e:
        print(f"SVD failed with error: {e}")
        print("Falling back to numpy SVD...")
        X_np = X_cpu.numpy()
        U_np, s_np, Vh_np = np.linalg.svd(X_np, full_matrices=False)
        U = torch.from_numpy(U_np)
        s = torch.from_numpy(s_np)
        Vh = torch.from_numpy(Vh_np)
    
    # Vh[-1] is the direction of least variance (smallest singular value)

    # Step 2: Generate canary feature vector
    # Scale to average norm of dataset
    sample_norms = torch.norm(X, dim=1)
    m = sample_norms.mean().item()
    x_p = Vh[-1] * m  # Vh[-1] has norm 1, scale to m
    print(f"Canary feature norm: {torch.norm(x_p).item():.6f}")

    # Step 3: Select target label that maximizes gradient norm
    print("Selecting optimal target label...")
    criterion = nn.CrossEntropyLoss()
    max_grad_norm = 0
    best_y = 0
    
    # Get device from model parameters
    model_device = next(model.parameters()).device
    x_p = x_p.float().to(model_device)  # Convert to float32 and move to model device
    
    # Reshape x_p back to original image shape if needed
    if len(X.shape) > 2:
        original_shape = X.shape[1:]  # (channels, height, width)
        x_p = x_p.view(*original_shape)
        print(f"Reshaped canary to image shape: {x_p.shape}")
    
    for y_candidate in range(out_dim):
        target = torch.tensor([y_candidate], dtype=torch.long, device=model_device)
        grad_norm = compute_gradient_norm(model, x_p, target, criterion)
        if grad_norm > max_grad_norm:
            max_grad_norm = grad_norm
            best_y = y_candidate
    
    y_p = best_y
    print(f"Selected target label: {y_p} with gradient norm: {max_grad_norm:.6f}")

    # Save canary
    canary_dict = {
        'feature': x_p.cpu(),
        'label': y_p
    }
    torch.save(canary_dict, args.output)
    print(f"Saved ClipBKD canary to {args.output}")
    print(f"Canary feature shape: {x_p.shape}")
    print(f"Canary label: {y_p}")


if __name__ == '__main__':
    main()
