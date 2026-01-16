# python generate_gradient_rotation_canary.py --data_name mnist --model_name cnn --num_iterations 100

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import copy
import numpy as np
from typing import Tuple, List

# Import existing utilities and models
from utils.data import load_data
from models import Models

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

# ============================================================================
# Training Functions
# ============================================================================

def train_one_epoch(model, dataset, device, lr=3):
    """Train model for one epoch using SGD"""
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # Simple batch processing
    batch_size = 32
    indices = torch.randperm(len(dataset))
    
    for i in range(0, len(dataset), batch_size):
        batch_indices = indices[i:i + batch_size]
        batch_x = torch.stack([dataset[j][0] for j in batch_indices]).to(device)
        batch_y = torch.tensor([dataset[j][1] for j in batch_indices]).to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()


def compute_per_sample_gradient(model, x, y, device):
    """
    Compute per-sample gradient for a single input.
    
    Returns:
        Flattened gradient vector
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    # Add batch dimension if needed
    if x.dim() == 3:
        x = x.unsqueeze(0)
    
    x = x.to(device)
    y = torch.tensor([y]).to(device)
    
    # Zero gradients
    model.zero_grad()
    
    # Forward pass
    output = model(x)
    loss = criterion(output, y)
    
    # Compute gradients w.r.t model parameters
    # We need create_graph=True so that the gradient itself is differentiable w.r.t x
    # This allows us to optimize x to maximize/minimize properties of the gradient (like norm or direction)
    params = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(loss, params, create_graph=True)
    
    # Flatten and concatenate all gradients
    grad_vector = torch.cat([g.view(-1) for g in grads])
    
    return grad_vector


def cosine_similarity(v1, v2):
    """Compute cosine similarity between two vectors"""
    return torch.dot(v1, v2) / (torch.norm(v1) * torch.norm(v2) + 1e-8)


def evaluate_model(model, dataset, device):
    """Evaluate model accuracy on dataset"""
    model.eval()
    correct = 0
    total = 0
    batch_size = 100
    
    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i + batch_size]
            batch_x = torch.stack([item[0] for item in batch]).to(device)
            batch_y = torch.tensor([item[1] for item in batch]).to(device)
            
            outputs = model(batch_x)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            
    return 100 * correct / total


# ============================================================================
# Main Canary Generation
# ============================================================================

def generate_gradient_rotation_canary(args, device):
    """Generate a gradient rotation canary"""
    
    print(f"Loading {args.data_name} dataset...")
    X, y, out_dim = load_data(args.data_name, n_df=500, split='train')
    D_train = list(zip(X, y))
    
    print(f"Initializing {args.model_name} model...")
    model_init = Models[args.model_name](X.shape, out_dim).to(device)
    
    if args.model_name == 'cnn':
        xavier_init_model(model_init)
    else:
        init_wideresnet(model_init)
    
    # Canary initialization
    print("Initializing canary...")
    input_shape = D_train[0][0].shape
    x_canary = torch.randn(input_shape, requires_grad=True, device=device)
    canary_optimizer = optim.Adam([x_canary], lr=0.01)
    
    print(f"Starting optimization for {args.num_iterations} iterations...")
    
    for iteration in range(args.num_iterations):
        # Simulate Epoch t (Training WITH Canary)
        model_t = copy.deepcopy(model_init)
        
        # Create temporary dataset with canary
        x_canary_detached = x_canary.detach().cpu()
        D_temp = D_train + [(x_canary_detached, args.y_target)]
        
        # Train for one epoch with canary
        train_one_epoch(model_t, D_temp, device)
        
        # Compute Gradient at Epoch t (g_t)
        g_t = compute_per_sample_gradient(model_t, x_canary, args.y_target, device)
        grad_norm_t = torch.norm(g_t, p=float('inf'))
        
        # Simulate Epoch t+1 (Training WITHOUT Canary)
        model_t1 = copy.deepcopy(model_t)
        
        # Train for one epoch without canary
        train_one_epoch(model_t1, D_train, device)
        
        # Compute Gradient at Epoch t+1 (g_{t+1})
        g_t1 = compute_per_sample_gradient(model_t1, x_canary, args.y_target, device)
        
        # Loss Calculation
        cos_sim = cosine_similarity(g_t, g_t1)
        loss = -grad_norm_t + 10.0 * cos_sim
        
        # Update Step
        canary_optimizer.zero_grad()
        loss.backward()
        canary_optimizer.step()
        
        # Clamp to [0, 1]
        with torch.no_grad():
            x_canary.clamp_(0, 1)
        
        if (iteration + 1) % 10 == 0 or iteration == 0:
            print(f"Iteration {iteration + 1}/{args.num_iterations}: "
                  f"Loss = {loss.item():.4f}, "
                  f"Grad Norm = {grad_norm_t.item():.4f}, "
                  f"Cos Sim = {cos_sim.item():.4f}")
    
    # Final Output
    print("\nOptimization complete!")
    
    # Calculate and print model utility
    print("Evaluating model utility on test set...")
    try:
        X_test, y_test, _ = load_data(args.data_name, n_df=1000, split='test')
        D_test = list(zip(X_test, y_test))
        if 'model_t1' in locals():
            acc = evaluate_model(model_t1, D_test, device)
            print(f"Final Model Test Accuracy: {acc:.2f}%")
        else:
            print("Model not defined (num_iterations=0?)")
    except Exception as e:
        print(f"Error evaluating model utility: {e}")

    x_canary_final = x_canary.detach().cpu()
    
    canary_dict = {
        'canary': x_canary_final,
        'audit_label': args.y_target
    }
    
    torch.save(canary_dict, args.output)
    print(f"Canary saved to: {args.output}")
    
    return canary_dict


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate a gradient rotation canary for privacy auditing'
    )
    
    parser.add_argument(
        '--data_name',
        type=str,
        default='mnist',
        help='Dataset name (default: mnist)'
    )
    
    parser.add_argument(
        '--model_name',
        type=str,
        default='cnn',
        help='Model architecture (default: cnn)'
    )
    
    parser.add_argument(
        '--y_target',
        type=int,
        default=0,
        help='Target class label for the canary (default: 0)'
    )
    
    parser.add_argument(
        '--num_iterations',
        type=int,
        default=100,
        help='Number of optimization iterations (default: 100)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='gradient_rotation_canary_2.pt',
        help='Output file path (default: gradient_rotation_canary_2.pt)'
    )
    
    args = parser.parse_args()
    
    # Automatically select device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Generate canary
    generate_gradient_rotation_canary(args, device)


if __name__ == '__main__':
    main()