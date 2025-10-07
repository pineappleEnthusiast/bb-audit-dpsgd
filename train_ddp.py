"""Distributed training script for DP-SGD auditing."""
import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import argparse
from parallel_audit_model import train_single_model, Models, xavier_init_model, init_wideresnet

def setup(rank, world_size):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """Clean up distributed training."""
    dist.destroy_process_group()

def train(rank, world_size, args):
    """Train a single model using DDP."""
    # Set up the process group
    setup(rank, world_size)
    
    # Set device
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    
    # Load data
    X, y, out_dim = load_data(args.data_name, args.n_df)
    
    # Initialize model
    if args.fixed_init:
        model = torch.load(args.fixed_init)
    else:
        model = Models[args.model_name](X.shape, out_dim=out_dim).to(device)
        if args.model_name == 'cnn':
            xavier_init_model(model)
        else:
            init_wideresnet(model)
    
    # Wrap model in DDP
    model = DDP(model, device_ids=[rank])
    
    # Train the model
    trained_model = train_single_model(
        model_name=args.model_name,
        X=X,
        y=y,
        X_target=None,  # Add your target data here
        y_target=None,  # Add your target labels here
        epsilon=args.epsilon,
        delta=args.delta,
        max_grad_norm=args.max_grad_norm,
        n_epochs=args.n_epochs,
        lr=args.lr,
        block_size=args.block_size,
        batch_size=args.batch_size,
        init_model=model.module if hasattr(model, 'module') else model,
        out_dim=out_dim,
        aug_mult=args.aug_mult,
        gradient_space_audit=args.target_type == 'gradient_space_canary',
        defense=args.defense,
        seed=args.seed + rank
    )
    
    # Save the model
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        model_path = os.path.join(args.output_dir, f'model_rank{rank}.pt')
        torch.save(trained_model.state_dict(), model_path)
    
    # Clean up
    cleanup()

def main():
    parser = argparse.ArgumentParser(description='Distributed DP-SGD Training')
    parser.add_argument('--data_name', type=str, default='mnist', help='Dataset name')
    parser.add_argument('--model_name', type=str, default='cnn', help='Model name')
    parser.add_argument('--n_df', type=int, default=0, help='Number of data points')
    parser.add_argument('--n_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--block_size', type=int, default=32, help='Block size for DP')
    parser.add_argument('--epsilon', type=float, default=None, help='DP epsilon')
    parser.add_argument('--delta', type=float, default=1e-5, help='DP delta')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Max gradient norm')
    parser.add_argument('--aug_mult', type=int, default=1, help='Augmentation multiplier')
    parser.add_argument('--defense', action='store_true', help='Use defense')
    parser.add_argument('--target_type', type=str, default='gradient_space_canary', help='Target type')
    parser.add_argument('--fixed_init', type=str, default=None, help='Path to fixed initialization')
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Get the number of GPUs
    world_size = torch.cuda.device_count()
    
    # Launch distributed training
    mp.spawn(
        train,
        args=(world_size, args),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    main()
