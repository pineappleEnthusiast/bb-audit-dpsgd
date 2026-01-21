
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import argparse
import time
from tqdm import tqdm

from models import Models
from utils.data import load_data

def train(args):
    # Set device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using MPS")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    # Load Data
    print(f"Loading {args.model_name} data...")
    # Use load_data utility. data_name='mnist'. 
    # n_df corresponds to number of samples. -1 means full dataset.
    # parallel_audit_model uses args.n_df, where 0 implies full dataset -> passes -1.
    n_examples = -1 if args.n_examples == 0 else args.n_examples
    
    # load_data(data_name, n_df, root='./', split='train')
    try:
        X, y, out_dim = load_data('mnist', n_examples)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print(f"Data loaded: X shape={X.shape}, y shape={y.shape}")
    print(f"Output dimension: {out_dim}")

    # Create Dataset and Loader
    dataset = TensorDataset(X, y)
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )

    # Initialize Model
    print(f"Initializing {args.model_name} model...")
    if args.model_name not in Models:
        print(f"Error: Model {args.model_name} not found in Models.")
        print(f"Available models: {list(Models.keys())}")
        return

    model_cls = Models[args.model_name]
    # Models normally take (in_shape, out_dim)
    try:
        model = model_cls(X.shape, out_dim).to(device)
    except Exception as e:
        print(f"Error initializing model: {e}")
        return

    # Optimizer and Loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    print("Starting training...")
    start_time = time.time()
    
    model.train()
    for epoch in range(args.epochs):
        epoch_start = time.time()
        total_loss = 0
        correct = 0
        total = 0
        
        # Use tqdm for progress bar
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}", unit="batch")
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # Update progress bar
            current_loss = total_loss / (batch_idx + 1)
            current_acc = 100. * correct / total
            pbar.set_postfix({'loss': f"{current_loss:.4f}", 'acc': f"{current_acc:.2f}%"})
        
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1} done in {epoch_time:.2f}s. Avg Loss: {total_loss/len(loader):.4f}, Acc: {100.*correct/total:.2f}%")

    total_time = time.time() - start_time
    print(f"Training finished in {total_time:.2f}s")
    
    # Save model
    if args.save_path:
        torch.save(model.state_dict(), args.save_path)
        print(f"Model saved to {args.save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Normal SGD Training for MNIST')
    parser.add_argument('--model_name', type=str, default='cnn', help='Model architecture to use (default: cnn)')
    parser.add_argument('--n_examples', type=int, default=0, help='Number of examples to use (0 for full dataset)')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of data loading workers')
    parser.add_argument('--save_path', type=str, default='mnist_sgd_model.pt', help='Path to save trained model')
    
    args = parser.parse_args()
    train(args)
