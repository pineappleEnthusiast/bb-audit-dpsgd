"""Normal SGD training loop for MNIST using parallel_audit_model utilities"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import argparse
import time
import torch.nn.functional as F

from models import Models
from utils.data import load_data


def xavier_init_model(model):
    """Initialize model using Xavier initialization"""
    def init_weights(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)
    model.apply(init_weights)


def create_soft_labels(y, num_classes, gamma=0.0):
    """
    Convert hard labels to soft labels with controlled entropy.
    
    Args:
        y: Hard labels (batch_size,)
        num_classes: Number of classes
        gamma: Entropy control parameter in [0, 1]
               - gamma=0: Hard labels (100% on ground-truth class)
               - gamma=1: Uniform distribution (1/num_classes on all classes)
               - gamma=0.8: Ground-truth gets ~20% probability
    
    Returns:
        Soft labels (batch_size, num_classes)
    
    Example:
        For gamma=0.8 and num_classes=10:
        - Ground-truth class: 0.2 (20%)
        - Other 9 classes: 0.8/9 ≈ 0.089 each (total 80%)
    """
    if gamma == 0.0:
        # Return one-hot encoded labels
        return F.one_hot(y, num_classes=num_classes).float()
    
    batch_size = y.size(0)
    soft_labels = torch.ones(batch_size, num_classes, device=y.device) * (gamma / num_classes)
    
    # Set ground-truth class probability
    ground_truth_prob = 1.0 - gamma + (gamma / num_classes)
    soft_labels.scatter_(1, y.unsqueeze(1), ground_truth_prob)
    
    return soft_labels


def test_model(model, X, y, batch_size=128, device='cuda'):
    """Test model accuracy"""
    model = model.to(device)
    X = X.to(device)
    y = y.to(device)

    test_loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=False)

    model.eval()
    acc = 0
    total = 0
    with torch.no_grad():
        for curr_X, curr_y in test_loader:
            curr_X = curr_X.to(device)
            curr_y = curr_y.to(device)
            curr_y_hat = torch.argmax(model(curr_X), dim=1)
            acc += torch.sum(curr_y_hat == curr_y).cpu().item()
            total += len(curr_y)

    model.train()
    return acc / total if total > 0 else 0.0


def train_model(model_name, X_train, y_train, X_test, y_test, n_epochs, lr, batch_size, 
                out_dim=10, device='cuda', num_workers=2, gamma=0.0):
    """
    Train a model using standard SGD (no DP).
    
    Args:
        model_name: Name of model architecture ('cnn', 'lr', 'mlp', etc.)
        X_train: Training data
        y_train: Training labels
        X_test: Test data
        y_test: Test labels
        n_epochs: Number of training epochs
        lr: Learning rate
        batch_size: Batch size for training
        out_dim: Output dimension (number of classes)
        device: Device to train on
        num_workers: Number of data loading workers
        gamma: Soft label entropy parameter (0=hard labels, 1=uniform)
    
    Returns:
        Trained model
    """
    # Set device
    device = torch.device(device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    
    # Initialize model
    print(f"Initializing {model_name} model...")
    model = Models[model_name](X_train.shape, out_dim=out_dim).to(device)
    
    # Xavier initialization for CNN
    if model_name == 'cnn':
        xavier_init_model(model)
    
    # Print soft label info
    if gamma > 0:
        ground_truth_prob = 1.0 - gamma + (gamma / out_dim)
        other_prob = gamma / out_dim
        print(f"Using soft labels with gamma={gamma:.2f}")
        print(f"  Ground-truth class probability: {ground_truth_prob:.4f} ({ground_truth_prob*100:.2f}%)")
        print(f"  Other class probability: {other_prob:.4f} ({other_prob*100:.2f}%) each")
    
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    # Create DataLoader
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True if device.type == 'cuda' else False,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False
    )
    
    print("Starting training...")
    for epoch in range(n_epochs):
        epoch_start = time.time()
        epoch_loss = 0
        n_batches = 0
        
        for batch_idx, (curr_X, curr_y) in enumerate(loader):
            curr_X, curr_y = curr_X.to(device, non_blocking=True), curr_y.to(device, non_blocking=True)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(curr_X)
            
            # Compute loss with soft or hard labels
            if gamma > 0:
                # Soft labels: use KL divergence
                soft_targets = create_soft_labels(curr_y, out_dim, gamma)
                log_probs = F.log_softmax(output, dim=1)
                loss = F.kl_div(log_probs, soft_targets, reduction='batchmean')
            else:
                # Hard labels: use cross-entropy
                loss = F.cross_entropy(output, curr_y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        # Compute accuracies
        train_acc = test_model(model, X_train, y_train, batch_size=batch_size, device=device)
        test_acc = test_model(model, X_test, y_test, batch_size=batch_size, device=device)
        
        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / n_batches if n_batches > 0 else 0
        
        print(f"Epoch {epoch+1}/{n_epochs} | Time: {epoch_time:.2f}s | "
              f"Loss: {avg_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Test Acc: {test_acc*100:.2f}%")
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Normal SGD Training for MNIST')
    parser.add_argument('--model_name', type=str, default='cnn', 
                        choices=list(Models.keys()), help='Model architecture')
    parser.add_argument('--n_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--n_train', type=int, default=0, 
                        help='Number of training examples (0 for full dataset)')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--save_path', type=str, default=None, help='Path to save trained model')
    parser.add_argument('--gamma', type=float, default=0.0, 
                        help='Soft label entropy parameter (0=hard labels, 0.8=20%% ground-truth prob, 1=uniform)')
    
    args = parser.parse_args()
    
    # Validate gamma
    if not (0.0 <= args.gamma <= 1.0):
        raise ValueError(f"gamma must be in [0, 1], got {args.gamma}")
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Load data
    print("Loading MNIST data...")
    n_train = -1 if args.n_train == 0 else args.n_train
    X_train, y_train, out_dim = load_data('mnist', n_train, split='train')
    X_test, y_test, _ = load_data('mnist', -1, split='test')
    
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Train model
    model = train_model(
        model_name=args.model_name,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        n_epochs=args.n_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        out_dim=out_dim,
        gamma=args.gamma,
        device=args.device,
        num_workers=args.num_workers
    )
    
    # Save model if requested
    if args.save_path:
        torch.save(model.state_dict(), args.save_path)
        print(f"Model saved to {args.save_path}")


if __name__ == "__main__":
    main()
