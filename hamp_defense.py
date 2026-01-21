"""Normal SGD training loop for MNIST using parallel_audit_model utilities"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import argparse
import time

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
                out_dim=10, device='cuda', num_workers=2):
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
    
    model.train()
    criterion = nn.CrossEntropyLoss()
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
            loss = criterion(output, curr_y)
            
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
    
    args = parser.parse_args()
    
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
        device=args.device,
        num_workers=args.num_workers
    )
    
    # Save model if requested
    if args.save_path:
        torch.save(model.state_dict(), args.save_path)
        print(f"Model saved to {args.save_path}")


if __name__ == "__main__":
    main()
