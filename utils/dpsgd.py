"""
Utility functions to execute DP-SGD
"""
import torch
import torch.distributed as dist
import numpy as np
from torch.func import functional_call, vmap, grad
import matplotlib.pyplot as plt
import threading
import copy
import pdb
from opacus.grad_sample import GradSampleModule

def preaugment_batch_vectorized(X, y, aug_fn, aug_mult):
    if aug_mult == 1:
        return aug_fn(X), y
    
    X_rep = X.repeat_interleave(aug_mult, dim=0)
    X_aug = aug_fn(X_rep)
    
    # Repeat labels: [B] -> [B * aug_mult]
    y_aug = y.repeat_interleave(aug_mult)
    
    return X_aug, y_aug


def average_grads_over_augmentations_optimized(ps_grads, batch_size, aug_mult):
    if aug_mult == 1:
        return ps_grads
    
    ps_grads_avg = {}
    for name, grad in ps_grads.items():
        param_dims = grad.shape[1:]
        grad_reshaped = grad.reshape(batch_size, aug_mult, *param_dims)
        ps_grads_avg[name] = grad_reshaped.mean(dim=1)
        
    return ps_grads_avg


# Drop-in replacement for your original functions
def preaugment_batch(X, y, aug_fn, aug_mult):
    """Original function signature maintained for compatibility"""
    return preaugment_batch_vectorized(X, y, aug_fn, aug_mult)


def average_grads_over_augmentations(ps_grads, batch_size, aug_mult):
    """Original function signature maintained for compatibility"""
    return average_grads_over_augmentations_optimized(ps_grads, batch_size, aug_mult)


def get_per_sample_grads(model, X, y, criterion):
    """Compute per-sample gradients"""
    # Check if model is DDP-wrapped
    is_ddp = hasattr(model, 'module')
    
    # Get model parameters, handling DDP case
    if is_ddp:
        # For DDP, we need to use the module's parameters but with the original names
        model_to_use = model.module
        # Create a mapping from original names to parameters
        param_mapping = {name.replace('module.', ''): param for name, param in model.named_parameters()}
    else:
        model_to_use = model
        param_mapping = dict(model.named_parameters())

    if isinstance(model_to_use, torch.nn.LSTM):
        # --- Use Opacus GradSampleModule path ---
        grad_model = GradSampleModule(model_to_use)
        grad_model.train()

        # Forward + backward pass with per-sample grads
        outputs = grad_model(X)
        loss = criterion(outputs, y)
        loss.backward()

        # Collect per-sample grads
        ps_grads = {}
        for name, p in grad_model.named_parameters():
            if hasattr(p, "grad_sample") and p.grad_sample is not None:
                ps_grads[name] = p.grad_sample.detach().clone()
                p.grad_sample = None  # clear to save memory
            else:
                ps_grads[name] = torch.zeros((X.size(0), *p.shape), device=p.device)

        return ps_grads
    else:
        # map of parameter names : parameter values (without module prefix)
        params = {k: v.detach() for k, v in model_to_use.named_parameters()}
        # map of buffer names : buffer values (without module prefix)
        buffers = {k: v.detach() for k, v in model_to_use.named_buffers()}

        def compute_loss(params, buffers, sample, target):
            batch = sample.unsqueeze(0)
            targets = target.unsqueeze(0)
            
            # Forward pass - no no_grad() here to allow gradient computation
            predictions = functional_call(model_to_use, (params, buffers), (batch,))
            loss = criterion(predictions, targets)
            return loss
        
        # Compute gradients
        ft_compute_grad = grad(compute_loss)
        ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0, 0))
        
        # Get gradients with consistent naming (without module prefix)
        ps_grads = ft_compute_sample_grad(params, buffers, X, y)
        
        return ps_grads

def get_per_sample_grad_norms(per_sample_grads):
    """Compute L2 norms of per-sample gradients"""
    return torch.vstack([
        curr_grad.flatten(start_dim=1).norm(2, dim=1)
        for curr_grad in per_sample_grads.values()
    ]).norm(2, dim=0)


def clip_per_sample_grads(per_sample_grads, max_grad_norm):
    """Clip per-sample gradients to clipping norm"""
    ps_grad_norms = get_per_sample_grad_norms(per_sample_grads)

    ps_grad_scales = 1 / torch.maximum(
        torch.ones_like(ps_grad_norms),
        ps_grad_norms / max_grad_norm
    )

    ps_grads_clipped = {
        # broadcast
        name: curr_grad * ps_grad_scales[(...,) + (None,) * (curr_grad.dim() - 1)]
        for name, curr_grad in per_sample_grads.items() 
    }

    ps_grad_norms_clipped = get_per_sample_grad_norms(ps_grads_clipped)

    return ps_grads_clipped, { 'before': ps_grad_norms.cpu().numpy(), 'after': ps_grad_norms_clipped.cpu().numpy() }



def clip_and_accum_grads_block(model, X, y, optimizer, criterion, max_grad_norm, device='cuda', aug_fn=None, aug_mult=1, 
                             is_gradient_space_canary=False, crafted_gradient=None, canary_local_idx=None):
    """
    Add aug_fn and aug_mult params to support augmentation multiplicity outside vmap.

    If aug_mult > 1, apply augmentation multiplicity outside, then average grads.
    """
    optimizer.zero_grad()
    
    # Check if model is DDP-wrapped
    is_ddp = hasattr(model, 'module')
    
    # Get the actual model (unwrapped if DDP)
    model_to_use = model.module if is_ddp else model
    
    # Get parameter names without 'module.' prefix
    param_names = [name.replace('module.', '') for name in model.state_dict().keys() 
                  if not name.startswith('_forward_hooks') and not name.startswith('_backward_hooks')]
    
    if len(X) == 0:
        # Initialize zero gradients with correct names
        ps_grads = {name: torch.zeros_like(param).unsqueeze(dim=0) 
                   for name, param in model_to_use.named_parameters()}
    else:
        # Compute per-sample gradients
        # Pre-augment outside vmap if aug_mult > 1
        if aug_mult > 1 and aug_fn is not None:
            X_aug, y_aug = preaugment_batch(X, y, aug_fn, aug_mult)
            X_aug = X_aug.to(device)
            y_aug = y_aug.to(device)

            ps_grads = get_per_sample_grads(model, X_aug, y_aug, criterion)
            ps_grads = average_grads_over_augmentations(ps_grads, batch_size=len(X), aug_mult=aug_mult)
        else:
            X = X.to(device)
            y = y.to(device)
            ps_grads = get_per_sample_grads(model, X, y, criterion)
        
        # Apply gradient-space audit after getting the gradients but before clipping
        if is_gradient_space_canary:
            # For the last sample in the block, replace its gradient with a crafted one
            for name in ps_grads.keys():
                # Replace the last sample's gradient with the crafted one
                ps_grads[name][canary_local_idx] = crafted_gradient[name]
            
    if max_grad_norm is not None:
        ps_grads_clipped, _ = clip_per_sample_grads(ps_grads, max_grad_norm)
    else:
        ps_grads_clipped = ps_grads

    with torch.no_grad():
        accum_grad_block = {name: grad.sum(dim=0) for name, grad in ps_grads_clipped.items()}

    # last_layer_name = list(model.net.named_modules())[-1][0]
    # last_w_name = 'net.' + last_layer_name + '.weight'
    # last_b_name = 'net.' + last_layer_name + '.bias'

    per_sample_flat_grads = torch.cat([g.view(g.shape[0], -1) for g in ps_grads.values()], dim=1)
    all_norms = torch.zeros_like(y, dtype=torch.float32)
    for k in range(10):
        k_last_layer_grads = per_sample_flat_grads[y == k]
        centered_k_last_layer_grads = k_last_layer_grads - k_last_layer_grads.mean(dim=0, keepdim=True)
        # take norm of each class
        centered_k_last_layer_norms = centered_k_last_layer_grads.norm(float('inf'), dim=1)
        all_norms[y == k] = centered_k_last_layer_norms

    # flat_last_weights = ps_grads[last_w_name].flatten(start_dim=1)
    # last_biases = ps_grads[last_b_name]
    # last_layer_grads = torch.cat((flat_last_weights, last_biases), dim=1)
    
    # all_norms = torch.zeros_like(y, dtype=torch.float32)
    # for k in range(10):
    #     # center each class
    #     k_last_layer_grads = last_layer_grads[y == k]
    #     centered_k_last_layer_grads = k_last_layer_grads - k_last_layer_grads.mean(dim=0, keepdim=True)
    #     # take norm of each class
    #     centered_k_last_layer_norms = centered_k_last_layer_grads.norm(2, dim=1)
    #     all_norms[y == k] = centered_k_last_layer_norms

    return accum_grad_block, None, all_norms.cpu().numpy(), None





def clip_and_accum_grads(model, X, y, optimizer, criterion, max_grad_norm,
                         block_size=1024, scores=None, device='cuda',
                         global_indices=None, aug_mult: int = 1, aug_fn=None,
                         world_size=1, rank=0, batch_size=None, drop_mask=None,
                         is_gradient_space_canary=False, crafted_gradient=None):
    """
    Clip and accumulate gradients in blocks with support for distributed training.
    
    Args:
        X: Input tensor
        y: Target tensor
        global_indices: Global indices of the current batch in the full dataset
        scores: Pre-allocated array to store scores for the entire dataset
        world_size: Number of processes in distributed training
        is_gradient_space_canary: Whether to apply gradient-space canary to the last sample
    """
    if scores is None:
        raise ValueError("scores array must be provided")
    
    if drop_mask is not None and len(drop_mask) != len(X):
        raise ValueError(f"drop_mask length ({len(drop_mask)}) must match X length ({len(X)})")
    
    
    # Get indices of non-dropped samples
    active_indices = torch.ones(len(X), dtype=torch.bool, device=device)
    if drop_mask is not None:
        active_indices = ~torch.tensor(drop_mask, device=device)
    
    # Filter out dropped samples
    X = X[active_indices]
    y = y[active_indices]
    global_indices = global_indices[active_indices]
    
    # Check if this is the last batch and we should apply gradient space canary
    apply_gradient_space_canary = is_gradient_space_canary and (global_indices == (len(scores) - 1)).any()
    
    if len(X) == 0:
        return None, scores
    
    # Process in blocks for memory efficiency
    accum_grad = None
    n_samples = len(X)
    
    for i in range(0, n_samples, block_size):
        # Get current block
        idx_block = slice(i, min(i + block_size, n_samples))
        curr_X = X[idx_block]
        curr_y = y[idx_block]
        curr_global_indices = global_indices[idx_block]
        
        # Skip if no samples in this block
        if len(curr_X) == 0:
            continue
            
        # Check if this block contains the last sample (canary)
        block_contains_canary = apply_gradient_space_canary and (curr_global_indices == (len(scores) - 1)).any()

        # Get the local index of the last sample in the current block
        if block_contains_canary:
            last_sample_local_idx = (curr_global_indices == (len(scores) - 1)).nonzero()[0].item()
        else:
            last_sample_local_idx = None
        
        # Compute per-block gradients with clipping
        accum_grad_block, _, last_layer_norms, _ = clip_and_accum_grads_block(
            model, curr_X, curr_y, optimizer, criterion, max_grad_norm,
            device=device, aug_mult=aug_mult, aug_fn=aug_fn,
            is_gradient_space_canary=block_contains_canary,
            crafted_gradient=crafted_gradient,
            canary_local_idx=last_sample_local_idx
        )
        # Accumulate gradients
        if accum_grad is None:
            accum_grad = accum_grad_block
        else:
            with torch.no_grad():
                for name in accum_grad:
                    accum_grad[name] += accum_grad_block[name]
        
        # Update scores for this block
        scores[curr_global_indices.cpu().numpy()] = last_layer_norms

    # idx_blocks is relative to current chunk
    # we want to map each chunk to global indices
    # we want to store scores at global indices position


    # if len(drop_mask) - 1 in active_global_indices:
    #     print('Canary in this minibatch')
    #     print(scores[np.where(active_global_indices.cpu().numpy() == (len(drop_mask) - 1))[0][0]], sorted(scores)[-5:])

    # k = 5

    # # gets top k indices in scores
    # topk_idx = np.argpartition(-scores, k)[:k]

    # # scores is local

    # topk_global_idx = active_global_indices[topk_idx]

    # if len(drop_mask) - 1 in topk_global_idx:
    #     print('Canary is getting dumped')
    #     exit()

    # Sum gradients across all processes
    if world_size > 1 and accum_grad is not None:
            # Synchronize gradients
            for name in accum_grad:
                dist.all_reduce(accum_grad[name], op=dist.ReduceOp.SUM)
    
    return accum_grad, scores



