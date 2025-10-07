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
from utils.parallel_dpsgd import clip_and_accum_grads, get_per_sample_grads, init_distributed
from utils.audit import compute_eps_lower_from_mia, compute_eps_lower_from_mia_given_t
from utils.clipbkd import craft_clipbkd, choose_worstcase_label

import gc
import torch.nn.functional as F
import torchvision.transforms.v2 as v2
from torch.utils.data import Dataset


class IndexedTensorDataset(Dataset):
    """A dataset that includes the index of each sample."""
    def __init__(self, *tensors):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        
    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors) + (index,)
        
    def __len__(self):
        return self.tensors[0].size(0)


os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def fgsm_attack(model, X, y, epsilon=0.1, max_iter=10, alpha=0.01):
    """
    Perform iterative FGSM attack on the input X to make it misclassified as target class y.
    
    Args:
        model: The model to attack
        X: Input tensor (1, C, H, W)
        y: Target class (tensor)
        epsilon: Maximum perturbation (default: 0.1)
        max_iter: Maximum number of iterations (default: 10)
        alpha: Step size for each iteration (default: 0.01)
        
    Returns:
        Adversarial example and number of iterations used
    """
    # Ensure model is in evaluation mode
    model.eval()
    
    # Make a copy of the input and enable gradient computation
    X_adv = X.clone().detach().requires_grad_(True)
    
    # Keep track of the best (most confident) adversarial example
    best_adv = X_adv.detach().clone()
    best_confidence = -float('inf')
    
    for i in range(max_iter):
        # Forward pass
        output = model(X_adv)
        
        # Check if attack succeeded
        _, predicted = torch.max(output, 1)
        if predicted != y:
            # If we've found a successful adversarial example, return it
            return X_adv.detach(), i + 1
            
        # Track the most confident adversarial example
        confidence = F.softmax(output, dim=1)[0, y].item()
        if confidence > best_confidence:
            best_confidence = confidence
            best_adv = X_adv.detach().clone()
        
        # Calculate loss (negative log likelihood for the target class)
        loss = F.cross_entropy(output, y)
        
        # Backward pass to get gradients
        model.zero_grad()
        loss.backward()
        
        # Get the sign of the gradients
        data_grad = X_adv.grad.data
        sign_data_grad = data_grad.sign()
        
        # Create perturbed image with step size alpha
        X_adv = X_adv.detach() + alpha * sign_data_grad
        
        # Project back to epsilon ball around original image
        delta = X_adv - X
        delta = torch.clamp(delta, -epsilon, epsilon)
        X_adv = torch.clamp(X + delta, 0, 1).detach().requires_grad_(True)
    
    # If we get here, the attack didn't succeed within max_iter
    # Return the best adversarial example found
    return best_adv, max_iter


class AugmentationFunction:
    def __init__(self, image_size=32, channels=3):
        self.base_transforms = v2.Compose([
            v2.RandomCrop(image_size, padding=4),
            v2.RandomHorizontalFlip(p=0.5),
        ])
    
    def __call__(self, x):
        return self.base_transforms(x)


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
                flat_grad[local_idx] = 10000000
                grad = flat_grad.view(info['shape'])
                
            crafted_grad[name] = grad.unsqueeze(0)  # Add batch dimension
        else:
            # For non-trainable parameters, set gradient to zero
            crafted_grad[name] = torch.zeros_like(param).unsqueeze(0)
    
    # Sanity check: print the norm of the flattened gradient
    flat_grad = torch.cat([grad.view(-1) for grad in crafted_grad.values()], dim=0)
    flat_grad_norm = flat_grad.norm()
    print(f"Flattened crafted gradient norm: {flat_grad_norm}")
    
    return crafted_grad


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
    """
    Train a single model with the given arguments and return the trained model and its index.
    This function is designed to be called in parallel.
    """
    # Unpack arguments
    (model_name, X, y, X_target, y_target, epsilon, delta, max_grad_norm, n_epochs, lr,
     block_size, batch_size, init_model, out_dim, aug_mult, rank, world_size,
     gradient_space_audit, crafted_gradient, defense, model_idx, device_id, seed) = model_args
    
    # Set device and random seed
    device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create a copy of the initial model if provided
    if init_model is not None:
        model = Models[model_name](X.shape, out_dim=out_dim).to(device)
        model.load_state_dict(init_model.state_dict())
    else:
        model = Models[model_name](X.shape, out_dim=out_dim).to(device)
        if model_name == 'cnn':
            xavier_init_model(model)
        else:
            init_wideresnet(model)
    
    # Train the model
    model = train_model(
        model_name=model_name,
        X=X,
        y=y,
        X_target=X_target,
        y_target=y_target,
        epsilon=epsilon,
        delta=delta,
        max_grad_norm=max_grad_norm,
        n_epochs=n_epochs,
        lr=lr,
        block_size=block_size,
        batch_size=batch_size,
        init_model=model,  # Use the initialized model
        out_dim=out_dim,
        aug_mult=aug_mult,
        rank=0,  # Single GPU training
        world_size=1,  # Single process
        gradient_space_audit=gradient_space_audit,
        crafted_gradient=crafted_gradient,
        defense=defense
    )
    
    return model, model_idx

def setup_ddp():
    """Initialize distributed training."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        # Initialize the process group
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
        
        # Set device
        device = torch.device(f'cuda:{local_rank}')
        
        return {
            'rank': rank,
            'world_size': world_size,
            'local_rank': local_rank,
            'device': device,
            'is_main_process': (rank == 0)
        }
    else:
        # Default to single-GPU mode if not using DDP
        return {
            'rank': 0,
            'world_size': 1,
            'local_rank': 0,
            'device': torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
            'is_main_process': True
        }


def cleanup_ddp():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def train_single_model(model_name, X, y, X_target, y_target, epsilon, delta, max_grad_norm, 
                     n_epochs, lr, block_size, batch_size, init_model=None, out_dim=10, aug_mult=1, 
                     gradient_space_audit=False, crafted_gradient=None, defense=False, seed=42, ddp_info=None):
    """Train a single model with the given parameters.
    
    Args:
        model_name: Name of the model to train
        X: Training data
        y: Training labels
        X_target: Target samples for auditing
        y_target: Target labels for auditing
        epsilon: Privacy budget (epsilon)
        delta: Privacy parameter (delta)
        max_grad_norm: Maximum gradient norm for clipping
        n_epochs: Number of training epochs
        lr: Learning rate
        block_size: Size of blocks for blockwise processing
        batch_size: Batch size for training
        init_model: Optional pre-initialized model
        out_dim: Output dimension of the model
        aug_mult: Multiplier for data augmentation
        gradient_space_audit: Whether to perform gradient space audit
        crafted_gradient: Pre-computed gradient for audit
        defense: Whether to use defense mechanism
        seed: Random seed for reproducibility
        
    Returns:
        Trained model
    """
    # Set device and random seed
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    # Initialize model
    if init_model is None:
        model = Models[model_name](X.shape, out_dim=out_dim).to(device)
        if model_name == 'cnn':
            xavier_init_model(model)
        else:
            init_wideresnet(model)
    else:
        model = copy.deepcopy(init_model).to(device)
    
    # Initialize ddp_info if not provided
    if ddp_info is None:
        ddp_info = {
            'world_size': 1,
            'rank': 0,
            'local_rank': 0,
            'device': device,
            'is_main_process': True
        }
    
    # Wrap model in DDP if using multiple processes
    if ddp_info['world_size'] > 1:
        model = DDP(model, device_ids=[ddp_info['local_rank']], output_device=ddp_info['local_rank'])
        if ddp_info.get('is_main_process', False):
            print("Using DDP with", ddp_info['world_size'], "processes")
    
    # Set model to training mode
    model.train()
    print(f"Model training mode: {model.training}")

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

    drop_mask = None
    assert block_size <= batch_size, "block_size must be smaller than batch_size"

    if len(X.shape) > 2:
        aug_fn = AugmentationFunction(X.shape[2], X.shape[1])
    else:
        aug_fn = None

    # Create Dataset + DataLoader
    dataset = IndexedTensorDataset(X, y)
    
    # Initialize scores array and drop mask for the entire dataset
    scores = np.zeros(len(dataset))
    drop_mask = np.zeros(len(dataset), dtype=bool)  # All samples active (not dropped) initially
    
    # Create the DataLoader
    # Only enable pin_memory if the data is on CPU
    pin_memory = X.device.type == 'cpu'
    
    sampler = torch.utils.data.RandomSampler(
        dataset,
        replacement=False,
        generator=torch.Generator().manual_seed(seed)
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        pin_memory=pin_memory,  # Only pin memory if data is on CPU
        num_workers=4,
        persistent_workers=True,
        drop_last=False,
        generator=torch.Generator().manual_seed(seed)
    )

    assert loader is not None, "Loader is None"
    
    for epoch in range(n_epochs):
        epoch_start = time.time()
        optimizer.zero_grad()
        print(f"Epoch: {epoch} (Active samples: {int((~drop_mask).sum())}/{len(drop_mask)})", end='', flush=True)

        for batch_idx, (curr_X, curr_y, global_indices) in enumerate(loader):
            print(f"Batch {batch_idx}: X shape: {curr_X.shape}, y shape: {curr_y.shape}, indices shape: {global_indices.shape}")
            # Move batch to device
            # Only use non_blocking if data is pinned (on CPU)
            non_blocking = pin_memory
            curr_X = curr_X.to(device, non_blocking=non_blocking)
            curr_y = curr_y.to(device, non_blocking=non_blocking)
            global_indices = global_indices.to(device, non_blocking=non_blocking)
            
            # Prepare batch_drop_mask if drop_mask is provided
            batch_drop_mask = None
            if drop_mask is not None:
                with torch.no_grad():
                    # Convert to tensor if it's a numpy array
                    if isinstance(drop_mask, np.ndarray):
                        drop_mask_tensor = torch.from_numpy(drop_mask).to(device=device, dtype=torch.bool)
                    else:
                        drop_mask_tensor = drop_mask.to(device=device, dtype=torch.bool)
                    # Index using the global indices
                    batch_drop_mask = drop_mask_tensor[global_indices]
            
            # Clip & accumulate gradients in memory-safe blocks
            print('DEBUG: Entering clip_and_accum_grads')
            curr_accumulated_gradients, scores = clip_and_accum_grads(
                model=model,
                X=curr_X, 
                y=curr_y, 
                optimizer=optimizer, 
                criterion=criterion,
                max_grad_norm=max_grad_norm, 
                drop_mask=batch_drop_mask,
                block_size=block_size,
                scores=scores,
                device=device,
                global_indices=global_indices,
                aug_mult=aug_mult, 
                aug_fn=aug_fn,
                is_gradient_space_canary=gradient_space_audit,
                crafted_gradient=crafted_gradient,
                model_type='in'  # or 'out' depending on your use case
            )
            print('DEBUG: Exiting clip_and_accum_grads')

            # Apply the accumulated gradients to the model parameters
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name not in curr_accumulated_gradients:
                        print(f"Warning: Parameter {name} not found in accumulated gradients")
                        continue
                        
                    # Get the accumulated gradient and move to device
                    grad = curr_accumulated_gradients[name].to(device)
                    
                    # Add DP noise if needed
                    if noise_multiplier > 0 and max_grad_norm is not None:
                        noise = noise_multiplier * max_grad_norm * torch.randn_like(grad)
                        grad.add_(noise)
                    
                    # Update the parameter's gradient
                    if param.grad is None:
                        param.grad = grad.clone()
                    else:
                        param.grad.copy_(grad)
            
            # Take an optimization step
            optimizer.step()
            optimizer.zero_grad()
        
        # Print epoch time
        epoch_time = time.time() - epoch_start
        print(f" | Time: {epoch_time:.2f}s")
        
        # Only perform defense-related operations if defense flag is True
        if defense:
            # Find top-k samples per class for gradient ascent
            k = 5  # Number of top samples per class
            top_k_grads = {}
            
            # Get unique classes
            unique_classes = torch.unique(y).cpu()
            
            # For each class, find top-k samples with highest scores
            for cls in unique_classes:
                # Get indices of samples in this class
                cls_indices = (y.cpu() == cls.item()).nonzero(as_tuple=True)[0]
                if len(cls_indices) == 0:
                    continue
                    
                # Get scores for this class and ensure it's a PyTorch tensor
                cls_scores = torch.tensor(scores[cls_indices.cpu().numpy()], device=y.device)
                
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
                if X.shape[0] - 1 in dropped_indices and not hasattr(train_single_model, '_canary_dropped'):
                    print(f"\n[INFO] Canary (index {X.shape[0]-1}) was dropped from the training set!", drop_mask[-1])
                    train_single_model._canary_dropped = True  # Mark that we've seen the canary drop
        
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
    

def train_model_wrapper(args, gpu_id=0):
    """Wrapper function to handle GPU device assignment and model training.
    
    Args:
        args: Tuple of arguments to pass to train_single_model
        gpu_id: ID of the GPU to use for this process
        
    Returns:
        Tuple of (trained_model, model_index)
    """
    import os
    import torch
    
    # Initialize distributed training
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    if world_size > 1:
        try:
            # Initialize distributed training using the provided function
            rank, world_size, device = init_distributed()
            print(f"[Rank {rank}/{world_size}] Initialized distributed training on device {device}")
        except Exception as e:
            print(f"[Rank {rank}] Error initializing distributed training: {e}")
            # Fall back to single-GPU training if distributed init fails
            world_size = 1
            rank = 0
            local_rank = 0
            device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
            print(f"Falling back to single-GPU training on device {device}")
    
    # Set CUDA device for this process
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Unpack arguments and add device information
        model_name, X, y, X_target, y_target, epsilon, delta, max_grad_norm, n_epochs, lr, \
        block_size, batch_size, init_model, out_dim, aug_mult, gradient_space_audit, \
        crafted_gradient, defense, seed = args
        
        # Move data to the correct device
        X = X.to(device)
        y = y.to(device)
        X_target = X_target.to(device) if X_target is not None else None
        y_target = y_target.to(device) if y_target is not None else None
        
        # Clone init_model to avoid sharing weights between processes
        if init_model is not None:
            init_model = copy.deepcopy(init_model).to(device)
        
        # Get DDP info if available
        ddp_info = {
            'world_size': int(os.environ.get('WORLD_SIZE', 1)),
            'rank': int(os.environ.get('RANK', 0)),
            'local_rank': int(os.environ.get('LOCAL_RANK', 0)),
            'device': device,
            'is_main_process': int(os.environ.get('RANK', 0)) == 0
        }
        
        # Call the training function
        model = train_single_model(
            model_name=model_name,
            X=X,
            y=y,
            X_target=X_target,
            y_target=y_target,
            epsilon=epsilon,
            delta=delta,
            max_grad_norm=max_grad_norm,
            n_epochs=n_epochs,
            lr=lr,
            block_size=block_size,
            batch_size=batch_size,
            init_model=init_model,
            out_dim=out_dim,
            aug_mult=aug_mult,
            gradient_space_audit=gradient_space_audit,
            crafted_gradient=crafted_gradient,
            defense=defense,
            seed=seed,
            ddp_info=ddp_info
        )
        
        # Get the model index from the last argument (seed)
        model_idx = seed - 42  # Since we started seed at 42 and incremented
        
        # Move model to CPU to avoid GPU memory issues when returning
        model = model.cpu()
        
        # Clean up distributed training if needed
        if world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()
            print(f"[Rank {rank}] Cleaned up process group")
        
        return model, model_idx
        
    except Exception as e:
        print(f"Error in train_model_wrapper: {str(e)}")
        raise


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
            init_distributed()  # Initialize distributed training
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
    print('Reproducibility enabled')
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_folder = f'{args.out}/{args.data_name}_{args.model_name}_eps{args.epsilon}'
    os.makedirs(out_folder, exist_ok=True)
    os.makedirs(f'{out_folder}/models', exist_ok=True)

    # load data (define D-)
    print('Loading data')
    if args.n_df == 1:
        # load single data point
        X_out, y_out, out_dim = load_data(args.data_name, 1)
    else:
        # since n_df is 0 by default, loads full dataset
        X_out, y_out, out_dim = load_data(args.data_name, args.n_df - 1)

    print('Initializing model')
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

    print('Crafting target data point')
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
    elif args.target_type == 'fgsm':
        print("Preparing FGSM attack by training a model on the available data...")
        
        # Create a new model for FGSM
        fgsm_model = Models[args.model_name](X_out.shape, out_dim=out_dim).to(device)
        if args.model_name == 'cnn':
            xavier_init_model(fgsm_model)
        else:
            init_wideresnet(fgsm_model)
        
        # Train the model using the existing train_model function
        print("Training FGSM model...")
        
        # Use train_model with DP disabled (delta=0, max_grad_norm=inf)
        fgsm_model = train_model(
            model_name=args.model_name,
            X=X_out,
            y=y_out,
            X_target=None,
            y_target=None,
            epsilon=None,  # No DP
            delta=None,    # No DP
            max_grad_norm=None,  # No gradient clipping
            n_epochs=args.n_epochs,
            lr=args.lr,
            block_size=args.block_size,
            batch_size=args.batch_size,
            init_model=fgsm_model,
            out_dim=out_dim,
            aug_mult=args.aug_mult,
            rank=rank,
            world_size=world_size,
            gradient_space_audit=False,
            defense=False
        )
        print("FGSM model training completed")
        
        # Get the last sample and its true label
        original_X = X_out[-1].unsqueeze(0).to(device)
        original_y = y_out[-1].unsqueeze(0).to(device)
        
        # Choose a target class different from the original
        num_classes = out_dim
        target_class = (original_y + 1) % num_classes  # Simple way to pick a different class
        
        print(f"Performing FGSM attack on sample (original class: {original_y.item()}, target class: {target_class.item()})")
        
        # Perform iterative FGSM attack
        print("Running iterative FGSM attack...")
        target_X, iters_used = fgsm_attack(
            fgsm_model, 
            original_X, 
            target_class, 
            epsilon=0.1,  # Maximum perturbation
            max_iter=20,  # Maximum iterations
            alpha=0.01    # Step size
        )
        target_y = target_class
        print(f"FGSM attack completed in {iters_used} iterations")
        
        # Move back to CPU if needed
        if not target_X.is_cpu:
            target_X = target_X.cpu()
        if not target_y.is_cpu:
            target_y = target_y.cpu()
            
        # Clean up
        del fgsm_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        print("FGSM attack completed")
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
    
    print('Training models')
    # train M on D and D-
    # resume from checkpoint
    worlds = [args.fit_world_only] if args.fit_world_only else ['in', 'out']
    models = {'in': [], 'out': []}
    outputs, losses, all_losses, train_set_accs, test_set_accs = resume_checkpoint(out_folder, args.fit_world_only, args.resume)
    
    # Create the crafted gradient once if doing gradient space audit
    crafted_grad = None
    if args.target_type == 'gradient_space_canary':
        print('Creating crafted gradient')
        # Create a temporary model to generate the gradient
        temp_model = Models[args.model_name](X_out.shape, out_dim=out_dim).to(device)
        if args.model_name == 'cnn':
            xavier_init_model(temp_model)
        else:
            init_wideresnet(temp_model)
        crafted_grad = craft_gradient(model=temp_model, device=device)
        del temp_model  # Clean up the temporary model

    # Determine number of GPUs to use
    num_gpus = torch.cuda.device_count()
    
    # Initialize a global seed counter
    curr_seed = 42  # Starting seed value
    
    for world in worlds:
        # Set dataset according to "world"
        curr_X, curr_y = (X_out, y_out) if world == 'out' else (X_in, y_in)
        
        # Check how many reps initially completed
        reps_completed = len(losses[world])
        
        # Calculate number of models to train in this world
        total_models = args.n_reps // 2
        models_to_train = total_models - reps_completed
        
        if models_to_train <= 0:
            print(f"Skipping {world} world - all models already trained")
            continue
            
        print(f"Training {models_to_train} models in {world} world using {num_gpus} GPUs")
        
        # Prepare model arguments for parallel training
        model_args_list = []
        
        for i in range(reps_completed, total_models):
            # Assign models to GPUs in a round-robin fashion
            gpu_id = i % num_gpus
            # Use the current seed and increment it for the next model
            seed = curr_seed
            curr_seed += 1
            
            model_args = (
                args.model_name,  # model_name
                curr_X,          # X
                curr_y,          # y
                target_X,        # X_target
                target_y,        # y_target
                args.epsilon,    # epsilon
                args.delta,      # delta
                args.max_grad_norm,  # max_grad_norm
                args.n_epochs,   # n_epochs
                args.lr,         # lr
                args.block_size, # block_size
                args.batch_size, # batch_size
                init_model,      # init_model
                out_dim,         # out_dim
                args.aug_mult,   # aug_mult
                args.target_type == 'gradient_space_canary' and world == 'in',  # gradient_space_audit
                crafted_grad if args.target_type == 'gradient_space_canary' and world == 'in' else None,  # crafted_gradient
                args.defense,    # defense
                seed             # seed
            )
            model_args_list.append(model_args)
        
        # Set the start method to 'spawn' for CUDA compatibility
        import multiprocessing as mp
        mp.set_start_method('spawn', force=True)
        
        # Train models in parallel using ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=num_gpus, mp_context=mp.get_context('spawn')) as executor:
            futures = []
            for i, model_args in enumerate(model_args_list):
                # Submit training tasks with the wrapper function
                try:
                    # Assign models to GPUs in a round-robin fashion
                    gpu_id = i % num_gpus
                    future = executor.submit(
                        train_model_wrapper,
                        model_args,
                        gpu_id=gpu_id
                    )
                    futures.append(future)
                except Exception as e:
                    print(f"Error submitting training task: {str(e)}")
                    raise
                
            # Process results as they complete
            for future in as_completed(futures):
                try:
                    model, model_idx = future.result()
                    rep = model_idx - reps_completed + 1
                    print(f"Completed model {rep}/{models_to_train} in {world} world")
                    
                    # Process the trained model (only on rank 0)
                    if rank == 0:
                        model.eval()
                        with torch.no_grad():
                            # Move target data to the model's device
                            device = next(model.parameters()).device
                            target_X_device = target_X.to(device)
                            target_y_device = target_y.to(device)
                            
                            # Get model output
                            output = model(target_X_device.unsqueeze(0))
                            outputs[world].append(output[0].cpu().numpy())
                            
                            # Calculate loss
                            if args.target_type == 'gradient_space_canary' and world == 'in' and crafted_grad is not None:
                                # Handle gradient space audit case
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
                                # Standard loss calculation
                                loss = -nn.CrossEntropyLoss()(output, target_y_device).item()
                                losses[world].append(loss)
                            
                            # Save checkpoint after each model
                            save_checkpoint(out_folder, outputs, losses, all_losses, train_set_accs, test_set_accs, args.fit_world_only)
                            
                            # Get test set accuracy for first 5 models
                            if model_idx < 5 and world == 'in':
                                if len(X_out) > 0:
                                    train_acc = test_model(model, X_in, y_in)
                                    train_set_accs.append(train_acc)
                                    print(f'Train set acc: {train_acc:.4f}')
                                
                                test_acc = test_model(model, X_test, y_test)
                                test_set_accs.append(test_acc)
                                print(f'Test set acc: {test_acc:.4f}')
                                
                                # Save checkpoint with updated accuracies
                                save_checkpoint(out_folder, outputs, losses, all_losses, train_set_accs, test_set_accs, args.fit_world_only)
                                
                except Exception as e:
                    print(f"Error training model: {str(e)}")
                    raise
            
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
                        final_params = {n: p.detach().clone().to(device) for n, p in model.named_parameters()}
                        init_params = {n: p.detach().clone().to(device) for n, p in init_model.named_parameters()}
                        
                        # Calculate cosine similarity between crafted gradient and parameter update
                        update = {n: final_params[n] - init_params[n] for n, p in final_params.items()}
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