"""HAMP Defense: Auditing soft label privacy using MIA"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import argparse
import time
import torch.nn.functional as F
import numpy as np
import dill

from models import Models
from utils.data import load_data
from utils.audit import compute_eps_lower_from_mia


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
    """
    if gamma == 0.0:
        return F.one_hot(y, num_classes=num_classes).float()
    
    batch_size = y.size(0)
    soft_labels = torch.ones(batch_size, num_classes, device=y.device) * (gamma / num_classes)
    
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


def compute_per_sample_losses(model, X, y, device, batch_size=256):
    """Compute per-sample losses for all samples"""
    model = model.to(device)
    X = X.to(device)
    y = y.to(device)

    loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=False)
    per_sample_losses = []

    model.eval()
    with torch.no_grad():
        for curr_X, curr_y in loader:
            curr_X = curr_X.to(device)
            curr_y = curr_y.to(device)
            logits = model(curr_X)
            batch_losses = F.cross_entropy(logits, curr_y, reduction='none')
            per_sample_losses.append(batch_losses.detach().cpu())

    model.train()
    return torch.cat(per_sample_losses, dim=0).numpy()


def train_model(model_name, X, y, n_epochs, lr, batch_size, out_dim, gamma, device, init_model=None):
    """
    Train a model using standard SGD with soft labels.
    
    Args:
        model_name: Name of model architecture
        X: Training data
        y: Training labels
        n_epochs: Number of training epochs
        lr: Learning rate
        batch_size: Batch size
        out_dim: Output dimension (number of classes)
        gamma: Soft label entropy parameter
        device: Device to train on
        init_model: Optional initial model weights
    
    Returns:
        Trained model
    """
    device = torch.device(device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    
    # Initialize model
    if init_model is None:
        model = Models[model_name](X.shape, out_dim=out_dim).to(device)
        if model_name == 'cnn':
            xavier_init_model(model)
    else:
        model = torch.nn.modules.utils._replicate_for_data_parallel(init_model).to(device)
    
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    # Create DataLoader
    dataset = TensorDataset(X, y)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True if device.type == 'cuda' else False,
        num_workers=0  # Simplified for single-GPU
    )
    
    for epoch in range(n_epochs):
        for curr_X, curr_y in loader:
            curr_X, curr_y = curr_X.to(device, non_blocking=True), curr_y.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            output = model(curr_X)
            
            # Compute loss with soft or hard labels
            if gamma > 0:
                soft_targets = create_soft_labels(curr_y, out_dim, gamma)
                log_probs = F.log_softmax(output, dim=1)
                loss = F.kl_div(log_probs, soft_targets, reduction='batchmean')
            else:
                loss = F.cross_entropy(output, curr_y)
            
            loss.backward()
            optimizer.step()
    
    return model


def save_checkpoint(out_folder, outputs, losses, train_set_accs, test_set_accs):
    """Save checkpoint"""
    os.makedirs(out_folder, exist_ok=True)
    
    random_state = {
        'np': np.random.get_state(),
        'torch': torch.random.get_rng_state()
    }
    dill.dump(random_state, open(f'{out_folder}/random_state.dill', 'wb'))
    
    np.save(f'{out_folder}/outputs_in.npy', outputs['in'])
    np.save(f'{out_folder}/outputs_out.npy', outputs['out'])
    np.save(f'{out_folder}/losses_in.npy', losses['in'])
    np.save(f'{out_folder}/losses_out.npy', losses['out'])
    np.save(f'{out_folder}/train_set_accs.npy', train_set_accs)
    np.save(f'{out_folder}/test_set_accs.npy', test_set_accs)


def init_run_state(out_folder):
    """Initialize fresh run state"""
    outputs = {'out': [], 'in': []}
    losses = {'out': [], 'in': []}
    train_set_accs = []
    test_set_accs = []
    
    os.makedirs(out_folder, exist_ok=True)
    save_checkpoint(out_folder, outputs, losses, train_set_accs, test_set_accs)
    
    return outputs, losses, train_set_accs, test_set_accs


def main():
    parser = argparse.ArgumentParser(description='HAMP Defense: Soft Label Privacy Auditing')
    parser.add_argument('--model_name', type=str, default='cnn', 
                        choices=list(Models.keys()), help='Model architecture')
    parser.add_argument('--n_reps', type=int, default=200, help='Number of model repetitions')
    parser.add_argument('--n_df', type=int, default=0, help='Dataset size (0 = full dataset)')
    parser.add_argument('--n_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--gamma', type=float, default=0.0, 
                        help='Soft label entropy parameter (0=hard labels, 0.8=20%% ground-truth prob, 1=uniform)')
    parser.add_argument('--alpha', type=float, default=0.05, help='Significance level for empirical epsilon')
    parser.add_argument('--delta', type=float, default=1e-5, help='Privacy parameter delta')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--out', type=str, default='exp_data/', help='Output folder')
    parser.add_argument('--fixed_init', action='store_true', help='Use fixed initialization')
    
    args = parser.parse_args()
    
    # Validate gamma
    if not (0.0 <= args.gamma <= 1.0):
        raise ValueError(f"gamma must be in [0, 1], got {args.gamma}")
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create output folder
    out_folder = f'{args.out}/mnist_{args.model_name}_gamma{args.gamma:.1f}'
    os.makedirs(out_folder, exist_ok=True)
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Load data
    print('Loading MNIST data...')
    n_df = -1 if args.n_df == 0 else args.n_df - 1
    X_out, y_out, out_dim = load_data('mnist', n_df, split='train')
    X_test, y_test, _ = load_data('mnist', -1, split='test')
    
    print(f'Train: {X_out.shape}, Test: {X_test.shape}')
    
    # Initialize model (optional)
    init_model = None
    if args.fixed_init:
        print('Initializing fixed model...')
        init_model = Models[args.model_name](X_out.shape, out_dim=out_dim)
        if args.model_name == 'cnn':
            xavier_init_model(init_model)
    
    # Craft blank canary
    print('Crafting blank canary...')
    target_X = torch.zeros_like(X_out[[0]])  # Blank image (all zeros)
    target_y = torch.tensor([9], dtype=torch.long)  # Label as class 9
    print(f'Canary: blank image with label {target_y.item()}')
    
    # Define two worlds
    X_in = torch.vstack((X_out[:-1], target_X))
    y_in = torch.cat((y_out[:-1], target_y))
    
    # Print soft label info
    if args.gamma > 0:
        ground_truth_prob = 1.0 - args.gamma + (args.gamma / out_dim)
        other_prob = args.gamma / out_dim
        print(f"\nUsing soft labels with gamma={args.gamma:.2f}")
        print(f"  Ground-truth class probability: {ground_truth_prob:.4f} ({ground_truth_prob*100:.2f}%)")
        print(f"  Other class probability: {other_prob:.4f} ({other_prob*100:.2f}%) each")
    else:
        print("\nUsing hard labels (no defense)")
    
    # Initialize run state
    outputs, losses, train_set_accs, test_set_accs = init_run_state(out_folder)
    
    print(f'\nTraining {args.n_reps} models in each world...')
    
    # Training loop
    for world in ['in', 'out']:
        curr_X, curr_y = (X_out, y_out) if world == 'out' else (X_in, y_in)
        print(f"\n{'='*60}")
        print(f"Training '{world}' world ({len(curr_X)} samples)")
        print(f"{'='*60}")
        
        for rep in range(args.n_reps):
            if (rep + 1) % 10 == 0:
                print(f"Rep {rep+1}/{args.n_reps}...")
            
            # Train model
            model = train_model(
                args.model_name, curr_X, curr_y, args.n_epochs, args.lr,
                args.batch_size, out_dim, args.gamma, device, init_model
            )
            
            # Compute canary output and loss
            model.eval()
            with torch.no_grad():
                target_X_device = target_X.to(device)
                target_y_device = target_y.to(device)
                
                output = model(target_X_device)
                loss = -F.cross_entropy(output, target_y_device).cpu().item()
                
                outputs[world].append(output[0].cpu().numpy())
                losses[world].append(loss)
            
            # Get test accuracy for first 5 reps
            if rep < 5 and world == 'in':
                train_acc = test_model(model, X_in, y_in, device=device)
                test_acc = test_model(model, X_test, y_test, device=device)
                train_set_accs.append(train_acc)
                test_set_accs.append(test_acc)
                print(f'  Rep {rep}: Train acc: {train_acc*100:.2f}%, Test acc: {test_acc*100:.2f}%')
            
            # Save checkpoint periodically
            if (rep + 1) % 50 == 0:
                save_checkpoint(out_folder, outputs, losses, train_set_accs, test_set_accs)
        
        # Convert to numpy arrays
        outputs[world] = np.array(outputs[world])
        losses[world] = np.array(losses[world])
    
    # Final save
    save_checkpoint(out_folder, outputs, losses, train_set_accs, test_set_accs)
    
    # Compute empirical epsilon
    print(f"\n{'='*60}")
    print("Computing empirical epsilon...")
    print(f"{'='*60}")
    
    mia_scores = np.concatenate([losses['in'], losses['out']])
    mia_labels = np.concatenate([np.ones_like(losses['in']), np.zeros_like(losses['out'])])
    
    max_t, emp_eps = compute_eps_lower_from_mia(
        mia_scores, mia_labels, args.alpha, args.delta, 'GDP', n_procs=1
    )
    
    # Save results
    np.save(f'{out_folder}/emp_eps_loss.npy', [emp_eps])
    np.save(f'{out_folder}/mia_scores.npy', mia_scores)
    np.save(f'{out_folder}/mia_labels.npy', mia_labels)
    
    # Print results
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f'Gamma: {args.gamma}')
    print(f'Empirical epsilon: {emp_eps:.4f}')
    if train_set_accs:
        print(f'Train set accuracy: {np.mean(train_set_accs) * 100:.2f}%')
    if test_set_accs:
        print(f'Test set accuracy: {np.mean(test_set_accs) * 100:.2f}%')
    print(f"\nResults saved to: {out_folder}")


if __name__ == "__main__":
    main()
