"""Auditing DP-SGD in black-box setting"""
import os
import sys
import time
import copy
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import os
import sys
import numpy as np
import argparse
from opacus.accountants.utils import get_noise_multiplier
import copy
from torch.utils.data import TensorDataset, DataLoader, Dataset
import time
import dill

import pdb

import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED

from models import Models
from models.wideresnet import WSConv2d
from utils.data import load_data
from utils.dpsgd import clip_and_accum_grads
from utils.audit import compute_eps_lower_from_mia, compute_eps_lower_from_mia_given_t
from utils.clipbkd import craft_clipbkd, choose_worstcase_label

import gc
import torch.nn.functional as F
import torchvision.transforms.v2 as v2

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


def craft_gradient(model, hot_index=None, device='cuda'):
    """
    Craft a 1-hot gradient vector that spans all parameters in the model.
    The gradient will have a single 10000 at the specified index when all parameters are flattened.
    If hot_index is None, defaults to the middle index of the total parameters.
    
    Args:
        model: The model for which to craft the gradient
        hot_index: Index at which to place the 1-hot value (10000.0) in the flattened parameter space.
                  If None, uses the middle index of the total parameters.
        device: Device on which to create the gradient tensors
        
    Returns:
        Dictionary with the same structure as model parameters containing the crafted gradients
    """
    # Get model parameters and calculate total number of elements
    params = {}
    total_elements = 0
    
    # First pass: calculate total number of elements and store parameter info
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_elements = param.numel()
            params[name] = {
                'param': param,
                'start_idx': total_elements,
                'end_idx': total_elements + num_elements,
                'shape': param.shape
            }
            total_elements += num_elements
    
    # Set default hot_index to middle if not provided
    if hot_index is None:
        hot_index = total_elements // 2 if total_elements > 0 else 0
    
    # Validate hot_index is within bounds
    if hot_index < 0 or (total_elements > 0 and hot_index >= total_elements):
        raise ValueError(f"hot_index {hot_index} is out of bounds for model with {total_elements} parameters")
    
    # Second pass: create the 1-hot gradient for each parameter
    crafted_grad = {}
    for name, info in params.items():
        param = info['param']
        if param.requires_grad:
            # Create zero gradient for this parameter
            grad = torch.zeros_like(param)
            
            # Check if the 1-hot index falls within this parameter's range
            if info['start_idx'] <= hot_index < info['end_idx']:
                # Calculate the local index within this parameter
                local_idx = hot_index - info['start_idx']
                # Flatten the gradient, set the 1-hot value, and reshape back
                flat_grad = grad.view(-1)
                flat_grad[local_idx] = 10000.0
                grad = flat_grad.view(info['shape'])
                
            crafted_grad[name] = grad.unsqueeze(0)  # Add batch dimension
        else:
            # For non-trainable parameters, set gradient to zero
            crafted_grad[name] = torch.zeros_like(param).unsqueeze(0)
    
    return crafted_grad


class AugmentationFunction:
    def __init__(self, image_size=32, channels=3):
        self.base_transforms = v2.Compose([
            v2.RandomCrop(image_size, padding=4),
            v2.RandomHorizontalFlip(p=0.5),
        ])
    
    def __call__(self, x):
        return self.base_transforms(x)


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


    # def init_weights(m):
    #     if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
    #         torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    #         if m.bias is not None:
    #             m.bias.data.zero_()
    # model.apply(init_weights)


def setup_device():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    return device, 1  # Return device and world_size=1 for compatibility


def setup(rank, world_size, local_rank, master_addr=None, master_port='12355'):
    """Initialize the distributed environment."""
    # Check if already initialized
    if dist.is_initialized():
        print(f'[Rank {rank}] Process group already initialized')
        return
        
    # Get master address from environment if not provided
    if master_addr is None:
        master_addr = os.environ.get('MASTER_ADDR', 'localhost')
    master_port = os.environ.get('MASTER_PORT', master_port)
    
    print(f'[Rank {rank}] Starting setup with master={master_addr}:{master_port}, world_size={world_size}, local_rank={local_rank}')
    
    # Set environment variables
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = str(master_port)
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['LOCAL_RANK'] = str(local_rank)
    
    # Print NCCL debug info
    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['NCCL_DEBUG_SUBSYS'] = 'INIT,ENV,NET'
    
    print(f'[Rank {rank}] Environment set, initializing process group...')
    
    try:
        # Initialize the process group
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            rank=rank,
            world_size=world_size,
            timeout=datetime.timedelta(seconds=60)  # Add timeout to prevent hanging
        )
        print(f'[Rank {rank}] Process group initialized successfully')
    except Exception as e:
        print(f'[Rank {rank}] Error initializing process group: {str(e)}')
        raise
    
    # Set device for this process
    try:
        torch.cuda.set_device(local_rank)
        print(f'[Rank {rank}] CUDA device set to {torch.cuda.get_device_name(local_rank)}')
    except Exception as e:
        print(f'[Rank {rank}] Error setting CUDA device: {str(e)}')
        raise


def cleanup():
    try:
        if dist.is_initialized():
            # Ensure all processes have completed before destroying the process group
            if dist.get_world_size() > 1:
                torch.cuda.synchronize()
                dist.barrier()
            dist.destroy_process_group()
            print(f'[Rank {dist.get_rank() if dist.is_initialized() else 0}] Successfully cleaned up process group')
    except Exception as e:
        print(f'[Rank {dist.get_rank() if dist.is_initialized() else 0}] Error during cleanup: {str(e)}')


class IndexedTensorDataset(Dataset):
    """A dataset that includes the index of each sample."""
    def __init__(self, *tensors):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        
    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors) + (index,)
        
    def __len__(self):
        return self.tensors[0].size(0)


class DDPModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x):
        return self.model(x)


def train_model(model_name, X, y, X_target, y_target, epsilon, delta, max_grad_norm, 
               n_epochs, lr, block_size, batch_size, init_model=None, out_dim=10, aug_mult=1, rank=0, world_size=1,
               gradient_space_audit=False, crafted_gradient=None, defense=False):
    
    # Initialize distributed training
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    # Setup the process groups
    if world_size > 1 and not dist.is_initialized():
        try:
            setup(rank, world_size, local_rank)
        except Exception as e:
            print(f"[Rank {rank}] Error initializing process group: {e}")
            world_size = 1  # Fall back to single process
            rank = 0
    
    # Set device for this process
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    
    if rank == 0:
        print(f"Training on {world_size} GPUs across {world_size // torch.cuda.device_count()} nodes")

    # Initialize model
    if init_model is None:
        model = Models[model_name](X.shape, out_dim=out_dim).to(device)
        if model_name == 'cnn':
            xavier_init_model(model)
        else:
            init_wideresnet(model)
    else:
        model = copy.deepcopy(init_model).to(device)
    
    # Verify model is in training mode
    model.train()
    if rank == 0:
        print(f"Model training mode: {model.training}")

    # Wrap model with DDP if using multiple processes
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        print(f"Training on {world_size} GPUs across {world_size // torch.cuda.device_count()} nodes")
        
        # Function to strip 'module.' prefix from parameter names for DDP
        def strip_module_prefix(state_dict):
            return {k.replace('module.', ''): v for k, v in state_dict.items()}

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # Set DP noise
    if epsilon is not None:
        noise_multiplier = get_noise_multiplier(
            target_epsilon=epsilon,
            target_delta=delta,
            sample_rate=batch_size / len(X),
            epochs=n_epochs,
            accountant='rdp'
        )
    else:
        noise_multiplier = 0

    drop_mask = None # torch.zeros(len(y), device=device)
    assert block_size <= batch_size, "block_size must be smaller than batch_size"

    aug_fn = AugmentationFunction(X.shape[2], X.shape[1])

    # Create Dataset + DataLoader with DDP support
    dataset = IndexedTensorDataset(X, y)
    
    # Initialize scores array and drop mask for the entire dataset
    scores = np.zeros(len(dataset))
    drop_mask = np.zeros(len(dataset), dtype=bool)  # All samples active (not dropped) initially
    
    per_gpu_batch_size = batch_size // world_size if world_size > 1 else batch_size
    if rank == 0:
        print(f"per GPU: {per_gpu_batch_size}")
    
    # Create a regular sampler for DDP
    if world_size > 1:
        sampler = torch.utils.data.DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True
        )
    else:
        sampler = torch.utils.data.RandomSampler(
            dataset,
            replacement=False,
            num_samples=None,
            generator=torch.Generator().manual_seed(0)
        )
    
    # Create the DataLoader
    loader = DataLoader(
        dataset,
        batch_size=per_gpu_batch_size,
        sampler=sampler,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(0)  # Initial seed
    )
    
    for epoch in range(n_epochs):
        epoch_start = time.time()
        
        # Update the sampler's epoch for shuffling in DDP
        if world_size > 1 and hasattr(sampler, 'set_epoch'):
            sampler.set_epoch(epoch)
            
        optimizer.zero_grad()
        print(f"Epoch: {epoch} (Active samples: {int((~drop_mask).sum())}/{len(drop_mask)})", end='', flush=True)

        for batch_idx, (curr_X, curr_y, global_indices) in enumerate(loader):
            # Move batch to device asynchronously
            curr_X, curr_y = curr_X.to(device, non_blocking=True), curr_y.to(device, non_blocking=True)
            global_indices = global_indices.to(device, non_blocking=True)
            
            # Clip & accumulate gradients in memory-safe blocks
            curr_accumulated_gradients, scores = clip_and_accum_grads(
                model.module if world_size > 1 else model,  # Unwrap DDP model
                curr_X, curr_y, optimizer, criterion,
                max_grad_norm, 
                drop_mask=drop_mask[global_indices.cpu().numpy()] if drop_mask is not None else None,
                block_size=block_size,
                scores=scores,
                device=device,
                global_indices=global_indices,
                aug_mult=aug_mult, 
                aug_fn=aug_fn,
                world_size=world_size, 
                rank=rank, 
                batch_size=batch_size,
                is_gradient_space_canary=gradient_space_audit,
            )

            # Apply the accumulated gradients to the model parameters
            with torch.no_grad():
                # Get parameter names without 'module.' prefix
                param_names = [name.replace('module.', '') for name, _ in model.named_parameters()]
                
                for name, param in model.named_parameters():
                    # Remove 'module.' prefix for DDP models
                    clean_name = name.replace('module.', '')
                    if clean_name not in curr_accumulated_gradients:
                        print(f"Warning: Parameter {clean_name} not found in accumulated gradients")
                        continue
                        
                    # Get the accumulated gradient and move to device
                    grad = curr_accumulated_gradients[clean_name].to(device)
                    
                    # Add DP noise if needed
                    if noise_multiplier > 0 and max_grad_norm is not None:
                        # Generate noise on rank 0 and broadcast to other processes
                        if world_size > 1:
                            if rank == 0:
                                noise = noise_multiplier * max_grad_norm * torch.randn_like(grad)
                                dist.broadcast(noise, src=0)
                            else:
                                noise = torch.zeros_like(grad)
                                dist.broadcast(noise, src=0)
                        else:
                            noise = noise_multiplier * max_grad_norm * torch.randn_like(grad)
                        grad.add_(noise)
                    
                    # Update the parameter's gradient
                    if param.grad is None:
                        param.grad = grad.clone()
                    else:
                        param.grad.copy_(grad)
            
            # Take an optimization step
            optimizer.step()
            
            # Clear gradients for next iteration
            optimizer.zero_grad()
        
        # Print epoch time
        epoch_time = time.time() - epoch_start
        print(f" | Time: {epoch_time:.2f}s")
        
        # Only perform defense-related operations if defense flag is True
        if defense:
            print('Defense')
            # Find top-k samples per class for gradient ascent
            k = 5  # Number of top samples per class
            top_k_grads = {}
            
            # Get unique classes
            unique_classes = torch.unique(y).cpu().numpy()
            
            # For each class, find top-k samples with highest scores
            for cls in unique_classes:
                # Get indices of samples in this class
                cls_indices = (y.cpu() == cls).nonzero(as_tuple=True)[0]
                if len(cls_indices) == 0:
                    continue
                    
                # Get scores for this class
                cls_scores = scores[cls_indices]
                
                # Get top-k indices within this class
                _, topk_indices = torch.topk(cls_scores, min(k, len(cls_scores)))
                topk_global_indices = cls_indices[topk_indices]
                
                # Compute gradients for these samples
                model.zero_grad()
                
                # Get the samples and their targets
                X_topk = X[topk_global_indices].to(device)
                y_topk = y[topk_global_indices].to(device)
                
                # Get per-sample gradients using the utility function
                ps_grads = get_per_sample_grads(model, X_topk, y_topk, criterion)
                
                # Store the gradients for later use
                top_k_grads[cls] = ps_grads
                
                # Mark these samples as dropped
                dropped_indices = topk_global_indices.cpu().numpy()
                drop_mask[dropped_indices] = True
                
                # Check if canary (last index) was dropped
                if X.shape[0] - 1 in dropped_indices and not hasattr(train_model, '_canary_dropped'):
                    print(f"\n[INFO] Canary (index {X.shape[0]-1}) was dropped from the training set!")
                    train_model._canary_dropped = True  # Mark that we've seen the canary drop
        
            # Perform gradient ascent step with learning rate (outside class loop)
            if top_k_grads:
                model.zero_grad()
                
                # Sum gradients across all classes
                sum_grads = None
                for grads in top_k_grads.values():
                    if sum_grads is None:
                        sum_grads = {k: v.sum(dim=0) for k, v in grads.items()}
                    else:
                        for k in grads:
                            sum_grads[k] = sum_grads.get(k, 0) + grads[k].sum(dim=0)
                
                # Apply gradient ascent
                for name, param in model.named_parameters():
                    if name in sum_grads:
                        if param.grad is None:
                            param.grad = -lr * sum_grads[name]  # Negative for ascent
                        else:
                            param.grad.add_(-lr * sum_grads[name])
                
                # Take the gradient ascent step
                optimizer.step()
                optimizer.zero_grad()
        
            # Update scores for the next epoch
            scores.fill(0)

    return model
    

def test_model(model, X, y, batch_size=128):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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


def save_checkpoint(out_folder, outputs, losses, all_losses, train_set_accs, test_set_accs, fit_world_only):
    """Save checkpoint"""
    # create folder if not exists
    os.makedirs(out_folder, exist_ok=True)

    # save random state
    random_state = {
        'np': np.random.get_state(),
        'torch': torch.random.get_rng_state()
    }
    dill.dump(random_state, open(f'{out_folder}/random_state.dill', 'wb'))

    # save intermediate values
    if fit_world_only:
        np.save(f'{out_folder}/outputs_{fit_world_only}.npy', outputs[fit_world_only])
        np.save(f'{out_folder}/losses_{fit_world_only}.npy', losses[fit_world_only])
        np.save(f'{out_folder}/all_losses_{fit_world_only}.npy', all_losses[fit_world_only])

        if fit_world_only == 'out':
            np.save(f'{out_folder}/train_set_accs.npy', train_set_accs)
            np.save(f'{out_folder}/test_set_accs.npy', test_set_accs)
    else:
        np.save(f'{out_folder}/outputs_in.npy', outputs['in'])
        np.save(f'{out_folder}/outputs_out.npy', outputs['out'])
        np.save(f'{out_folder}/train_set_accs.npy', train_set_accs)
        np.save(f'{out_folder}/test_set_accs.npy', test_set_accs)
        np.save(f'{out_folder}/losses_in.npy', losses['in'])
        np.save(f'{out_folder}/losses_out.npy', losses['out'])
        np.save(f'{out_folder}/all_losses_in.npy', all_losses['in'])
        np.save(f'{out_folder}/all_losses_out.npy', all_losses['out'])


def resume_checkpoint(out_folder, fit_world_only, resume):
    """Load checkpoint if resume is set to True and previous checkpoint exists, else create new empty checkpoint"""
    outputs = {'out': [], 'in': []}
    losses = {'out': [], 'in': []}
    all_losses = {'in': [], 'out': []}
    train_set_accs = []
    test_set_accs = []

    if os.path.exists(out_folder) and resume:
        # if folder exists and resume is set to true load previous values
        random_state = dill.load(open(f'{out_folder}/random_state.dill', 'rb'))
        np.random.set_state(random_state['np'])
        torch.random.set_rng_state(random_state['torch'])

        if fit_world_only:
            outputs[fit_world_only] = np.load(f'{out_folder}/outputs_{fit_world_only}.npy').tolist()
            losses[fit_world_only] = np.load(f'{out_folder}/losses_{fit_world_only}.npy').tolist()
            all_losses[fit_world_only] = np.load(f'{out_folder}/all_losses_{fit_world_only}.npy').tolist()

            if fit_world_only == 'out':
                train_set_accs = np.load(f'{out_folder}/train_set_accs.npy').tolist()
                test_set_accs = np.load(f'{out_folder}/test_set_accs.npy').tolist()
        else:
            outputs['in'] = np.load(f'{out_folder}/outputs_in.npy').tolist()
            outputs['out'] = np.load(f'{out_folder}/outputs_out.npy').tolist()
            train_set_accs = np.load(f'{out_folder}/train_set_accs.npy').tolist()
            test_set_accs = np.load(f'{out_folder}/test_set_accs.npy').tolist()
            losses['in'] = np.load(f'{out_folder}/losses_in.npy').tolist()
            losses['out'] = np.load(f'{out_folder}/losses_out.npy').tolist()
            all_losses['in'] = np.load(f'{out_folder}/all_losses_in.npy').tolist()
            all_losses['out'] = np.load(f'{out_folder}/all_losses_out.npy').tolist()
    else:
        # create folder and dump initial values in
        os.makedirs(out_folder, exist_ok=True)
        save_checkpoint(out_folder, outputs, losses, all_losses, train_set_accs, test_set_accs, fit_world_only)
    
    return outputs, losses, all_losses, train_set_accs, test_set_accs


def main():
    # Parse command line arguments first
    parser = argparse.ArgumentParser()
    
    # Initialize distributed training
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    print(f'[Rank {rank}] Starting main process with world_size={world_size}, local_rank={local_rank}')
    
    # Print process info for all ranks
    print(f"[Rank {rank}] World size: {world_size}, Local rank: {local_rank}")
    print(f"[Rank {rank}] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[Rank {rank}] CUDA device count: {torch.cuda.device_count()}")
    
    # Initialize distributed training if needed
    if world_size > 1:
        print(f'[Rank {rank}] Initializing distributed training...')
        try:
            setup(rank, world_size, local_rank)
            print(f'[Rank {rank}] Distributed training initialized successfully')
        except Exception as e:
            print(f'[Rank {rank}] Failed to initialize distributed training: {str(e)}')
            raise
    
    # Set device for this process
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    
    try:
        if rank == 0:
            print(f"Training with {world_size} GPUs")
            
        # Create parser with allow_abbrev=False to avoid conflicts with custom argument handling
        parser = argparse.ArgumentParser(allow_abbrev=False)
        
        # Handle both --local_rank and --local-rank for compatibility
        for arg in sys.argv[1:]:
            if arg.startswith('--local-rank'):
                # Convert --local-rank to --local_rank for consistency
                sys.argv[sys.argv.index(arg)] = '--local_rank' + arg[12:]
        
        # Add local_rank argument
        parser.add_argument('--local_rank', type=int, default=0,
                         help='Local rank for distributed training')
        parser.add_argument('--data_name', type=str, default='mnist', help='dataset to use (mnist, cifar10, cifar100)')
        parser.add_argument('--model_name', type=str, default='lr', choices=list(Models.keys()), help='model to audit')
        parser.add_argument('--n_reps', type=int, default=200, help='number of models')
        parser.add_argument('--n_df', type=int, default=0, help='|D| (0 => use full dataset)')
        parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs to train for')
        parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
        parser.add_argument('--max_grad_norm', type=float, default=1, help='gradient clipping norm')
        parser.add_argument('--epsilon', type=float, default=None, help='privacy parameter, epsilon')
        parser.add_argument('--delta', type=float, default=1e-5, help='privacy parameter, delta')
        parser.add_argument('--target_type', type=str, default='blank', help='sample to use as target (blank, clipbkd, badnets, or path to target sample)')
        parser.add_argument('--blank_alpha', type=float, default=0.0, help='interpolation factor for blank target (0.0 = fully blank, 1.0 = fully label 9 image)')
        parser.add_argument('--seed', type=int, default=0, help='seed for reproducibility')
        parser.add_argument('--out', type=str, default='exp_data/', help='folder to write results to')
        parser.add_argument('--fixed_init', type=str, nargs='?', default=None, const='', help='initialize all models to the same weights (if path provided, weights loaded from path (worst-case), else fix to some randomly chosen weights)')
        parser.add_argument('--block_size', type=int, help='process samples within a batch in blocks to conserve GPU space')
        parser.add_argument('--batch_size', type=int, help='batch size for training')
        parser.add_argument('--resume', action='store_true', help='skip experiment if results are present')
        parser.add_argument('--fit_world_only', type=str, default=None, choices=['in', 'out'], help='just fit models in world and calculate losses')
        parser.add_argument('--alpha', type=float, default=0.05, help='significance level for empirical eps estimation')
        parser.add_argument('--badnets_label', type=int, default=-1, help='assign badnets poison this label')

        # Options for Debugging
        parser.add_argument('--view_badnets', action='store_true')
        parser.add_argument('--store_canary_rank', action='store_true')
        parser.add_argument('--holdout_audit', action='store_true')
        
        # Gradient-space audit options
        parser.add_argument('--target_class', type=int, default=0,
                          help='Target class for gradient-space audit')


        # Options for Forgetting Canary Candidates
        parser.add_argument('--defense', action='store_true', help='use filtering defense during audit')
        parser.add_argument('--aug_mult', type=int, default=1, help='augmentation multiplier (default: 1)')

        args = parser.parse_args()
        if args.max_grad_norm == -1: 
            args.max_grad_norm = None
            
    except Exception as e:
        print(f"Error in main: {str(e)}")
        if world_size > 1:
            cleanup()
        raise

    # reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_folder = f'{args.out}/{args.data_name}_{args.model_name}_eps{args.epsilon}'
    os.makedirs(out_folder, exist_ok=True)
    os.makedirs(f'{out_folder}/models', exist_ok=True)

    # load data (define D-)
    if args.n_df == 1:
        # load single data point
        X_out, y_out, out_dim = load_data(args.data_name, 1)
    else:
        # since n_df is 0 by default, loads full dataset
        X_out, y_out, out_dim = load_data(args.data_name, args.n_df - 1)

    init_model = None
    if args.fixed_init is not None:
        init_model = Models[args.model_name](X_out.shape, out_dim=out_dim)

        if args.fixed_init == '':
            # initialize model (average-case)
            if args.model_name == 'cnn':
                xavier_init_model(init_model)
            else:
                init_wideresnet(init_model)
        else:
            # load weights from path (worst-case)
            init_model.load_state_dict(torch.load(args.fixed_init))
            # don't train on the first half of the dataset
            X_out, y_out = X_out[len(X_out) // 2:], y_out[len(y_out) // 2:]

    # check for data_names + target_types that don't match
    if args.data_name == 'mnist':
        pass # compatible with all canaries
    elif args.data_name == 'cifar10':
        pass # compatible with all canaries
    elif args.data_name == 'cifar100':
        pass # compatible with all canaries
    elif args.data_name == 'purchase':
        # not compatible with badnets or clipbkd
        if args.target_type == 'badnets' or args.target_type == 'clipbkd':
            print("Warning: canary type does not support tabular data.")

    # craft target data point (x_T, y_T)
    if args.target_type == 'gradient_space_canary':
        # For gradient space canary, we don't modify the dataset
        # The last sample will be used as the canary
        target_X = X_out[-1].unsqueeze(0)  # Keep the last sample as target
        target_y = y_out[-1].unsqueeze(0)
        if rank == 0:
            print("Using gradient-space canary (last sample in dataset)")
    elif args.target_type == 'blank':
        # blank sample with optional interpolation
        blank_img = torch.zeros_like(X_out[[0]])
        if args.blank_alpha > 0:
            # Find first image with label 9 for interpolation
            label_9_indices = (y_out == 9).nonzero(as_tuple=True)[0]
            if len(label_9_indices) > 0:
                label_9_img = X_out[label_9_indices[0]].unsqueeze(0)
                target_X = (1 - args.blank_alpha) * blank_img + args.blank_alpha * label_9_img
            else:
                print("Warning: No label 9 image found for interpolation, using pure blank image")
                target_X = blank_img
        else:
            target_X = blank_img
        target_y = torch.from_numpy(np.array([9]))
    elif args.target_type == 'badnets':
        target_X = X_out[-1]
        print('Original Label:', y_out[-1])
        target_y = torch.tensor(args.badnets_label)
        target_X[:, -4:, -4:] = torch.max(target_X)

        target_X = target_X.unsqueeze(0)
        target_y = target_y.unsqueeze(0)

        if args.view_badnets:
            plt.imshow(target_X.squeeze().numpy(), cmap='gray')
            plt.savefig(f'badnets_{args.badnets_label}.png')

    elif args.target_type == 'sanity_check':
        target_X = X_out[-1].unsqueeze(0)
        target_y = y_out[-1].unsqueeze(0)

    elif args.target_type == 'clipbkd':
        # ClipBKD sample
        target_X, target_y = craft_clipbkd(X_out, init_model)
    elif os.path.exists(args.target_type):
        # pre-crafted target sample
        target_X = torch.from_numpy(np.load(args.target_type))
        if init_model is not None:
            target_y =  choose_worstcase_label(init_model, target_X)
        else:
            target_y = torch.from_numpy(np.array([9]))
    else:
        raise Exception(f'Target {args.target_type} not found')

    # define D = D- U {(x_T, y_T)}
    X_in, y_in = torch.vstack((X_out[:-1], target_X)), torch.cat((y_out[:-1], target_y))
    X_test, y_test, _ = load_data(args.data_name, None, split='test')
    
    # train M on D and D-
    # resume from checkpoint
    worlds = [args.fit_world_only] if args.fit_world_only else ['in', 'out']
    models = {'in': [], 'out': []}
    outputs, losses, all_losses, train_set_accs, test_set_accs = resume_checkpoint(out_folder, args.fit_world_only, args.resume)
    
    # Create the crafted gradient once if doing gradient space audit
    crafted_grad = None
    if args.target_type == 'gradient_space_canary':
        # Create a temporary model to generate the gradient
        temp_model = Models[args.model_name](X_out.shape, out_dim=out_dim).to(device)
        if args.model_name == 'cnn':
            xavier_init_model(temp_model)
        else:
            init_wideresnet(temp_model)
        crafted_grad = craft_gradient(model=temp_model, device=device)
        del temp_model  # Clean up the temporary model

    for world in worlds:
        # set dataset according to "world"
        curr_X, curr_y = (X_out, y_out) if world == 'out' else (X_in, y_in)

        # check how many reps initially completed
        reps_completed = len(losses[world])

        # Simple loop with print for progress
        for rep in range(reps_completed, args.n_reps // 2):
            if rank == 0:
                print(f"Rep {rep + 1}/{(args.n_reps // 2)}")
            
            # Train model on all ranks
            model = train_model(args.model_name, 
                                            curr_X, 
                                            curr_y, 
                                            target_X, 
                                            target_y, 
                                            args.epsilon, 
                                            args.delta,
                                            args.max_grad_norm, 
                                            args.n_epochs, 
                                            args.lr, 
                                            args.block_size, 
                                            args.batch_size,
                                            init_model=init_model,
                                            out_dim=out_dim, 
                                            defense=args.defense,
                                            aug_mult=args.aug_mult,
                                            rank=rank,
                                            world_size=world_size,
                                            gradient_space_audit=args.target_type == 'gradient_space_canary' and world == 'in',
                                            crafted_gradient=crafted_grad if args.target_type == 'gradient_space_canary' and world == 'in' else None)
            
            # Only rank 0 processes the rest
            if rank == 0:
                model.eval()
                with torch.no_grad():
                    # Ensure target data is on the same device as the model
                    device = next(model.parameters()).device
                    target_X_device = target_X.to(device)
                    target_y_device = target_y.to(device)
                    
                    output = model(target_X_device)
                    outputs[world].append(output[0].cpu().numpy())
                    
                    if args.target_type == 'gradient_space_canary' and world == 'in' and crafted_grad is not None:
                        # Calculate parameter update
                        final_params = {n: p.detach().clone() for n, p in model.named_parameters()}
                        init_params = {n: p.detach().clone() for n, p in init_model.named_parameters()}
                        
                        # Calculate cosine similarity between crafted gradient and parameter update
                        update = {n: final_params[n] - init_params[n] for n in final_params}
                        flat_crafted_grad = torch.cat([g.view(-1) for g in crafted_grad.values()])
                        flat_update = torch.cat([p.view(-1) for p in update.values()])
                        
                        # Normalize vectors for cosine similarity
                        flat_crafted_grad = flat_crafted_grad / (flat_crafted_grad.norm() + 1e-10)
                        flat_update = flat_update / (flat_update.norm() + 1e-10)
                        
                        cos_sim = (flat_crafted_grad * flat_update).sum().item()
                        losses[world].append(cos_sim)
                    else:
                        # Original loss calculation for non-gradient canary cases
                        losses[world].append(-nn.CrossEntropyLoss()(output, target_y_device).cpu().item())
            
            # Synchronize all processes after processing
            if world_size > 1:
                torch.distributed.barrier()
            
            # Save checkpoint after each rep (only rank 0)
            if rank == 0:
                save_checkpoint(out_folder, outputs, losses, all_losses, train_set_accs, test_set_accs, args.fit_world_only)
                
                # get test set accuracy from first 5 reps
                if rep < 5 and world == 'in':
                    if len(X_out) > 0:
                        train_set_accs.append(test_model(model, X_in, y_in))
                        print('Train set acc:', train_set_accs[-1])
                    test_set_accs.append(test_model(model, X_test, y_test))
                    print('Test set acc:', test_set_accs[-1])

                # save checkpoint
                save_checkpoint(out_folder, outputs, losses, all_losses, train_set_accs, test_set_accs, args.fit_world_only)
        
        if rank == 0:
            outputs[world] = np.array(outputs[world])

    if rank == 0:
        if not args.fit_world_only:
            def audit_canary(losses, args):        
                k = len(losses['in'])
                t_losses = {'in': None, 'out': None}
                holdout_losses = {'in': None, 'out': None}

                if args.holdout_audit:
                    k = len(losses['in']) // 2
                
                t_losses['in'] = losses['in'][:k]
                t_losses['out'] = losses['out'][:k]
                holdout_losses['in'] = losses['in'][k:]
                holdout_losses['out'] = losses['out'][k:]

                # calculate empirical epsilon using GDP
                mia_scores = np.concatenate([t_losses['in'], t_losses['out']])
                mia_labels = np.concatenate([np.ones_like(t_losses['in']), np.zeros_like(t_losses['out'])])

                max_t, emp_eps_loss = compute_eps_lower_from_mia(mia_scores, mia_labels, args.alpha, args.delta, 'GDP', n_procs=1)

                if args.holdout_audit:
                    emp_eps_loss = compute_eps_lower_from_mia_given_t(np.concatenate(
                        [holdout_losses['in'], holdout_losses['out']]), 
                        np.concatenate([np.ones_like(holdout_losses['in']), np.zeros_like(holdout_losses['out'])]), 
                        args.alpha, 
                        args.delta, 
                        max_t, 
                        'GDP')
                
                return emp_eps_loss, mia_scores, mia_labels
            
            emp_eps_loss, mia_scores, mia_labels = audit_canary(losses, args)

            np.save(f'{out_folder}/emp_eps_loss.npy', [emp_eps_loss])
            np.save(f'{out_folder}/mia_scores.npy', mia_scores)
            np.save(f'{out_folder}/mia_labels.npy', mia_labels)
        
            print(f'Theoretical eps: {args.epsilon}')
            print(f'Empirical eps: {emp_eps_loss}')

        print(f'Train set accuracy: {np.mean(train_set_accs) * 100:.3f}%')
        print(f'Test set accuracy: {np.mean(test_set_accs) * 100:.3f}%')

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nInterrupted by user')
    except Exception as e:
        print(f'\nError in main: {str(e)}')
    finally:
        # Ensure cleanup is always called, even if there's an error
        if 'dist' in globals() and dist.is_initialized():
            cleanup()