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

    # map of parameter names : parameter values
    params = {k: v.detach() for k, v in model.named_parameters()}
    # map of buffer names : buffer balues
    buffers = {k: v.detach() for k, v in model.named_buffers()}

    def compute_loss(params, buffers, sample, target):
        batch = sample.unsqueeze(0)
        targets = target.unsqueeze(0)

        predictions = functional_call(model, (params, buffers), (batch,))
        loss = criterion(predictions, targets)
        return loss
    
    ft_compute_grad = grad(compute_loss)
    ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0, 0))
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




def clip_and_accum_grads_block(model, X, y, optimizer, criterion, max_grad_norm, device='cuda', aug_fn=None, aug_mult=1):
    """
    Add aug_fn and aug_mult params to support augmentation multiplicity outside vmap.

    If aug_mult > 1, apply augmentation multiplicity outside, then average grads.
    """
    optimizer.zero_grad()

    if len(X) == 0:
        ps_grads = {name: torch.zeros_like(param).unsqueeze(dim=0) for name, param in model.named_parameters()}
    else:
        # Pre-augment outside vmap if aug_mult > 1
        if aug_mult > 1 and aug_fn is not None:
            X_aug, y_aug = preaugment_batch(X, y, aug_fn, aug_mult)
            X_aug = X_aug.to(device)
            y_aug = y_aug.to(device)

            ps_grads_aug = get_per_sample_grads(model, X_aug, y_aug, criterion)
            ps_grads = average_grads_over_augmentations(ps_grads_aug, batch_size=len(X), aug_mult=aug_mult)

        else:
            X = X.to(device)
            y = y.to(device)
            ps_grads = get_per_sample_grads(model, X, y, criterion)

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

    return accum_grad_block, _, all_norms.cpu().numpy(), None





def clip_and_accum_grads(model, X, y, optimizer, criterion, max_grad_norm,
                         block_size=1024, drop_mask=None, scores=None, device='cuda',
                         original_indices=None, aug_mult: int = 1, aug_fn=None,
                         world_size=1, rank=0, batch_size=None):
    """
    Clip and accumulate gradients in blocks with support for distributed training.
    
    Args:
        world_size: Number of processes in distributed training
        rank: Rank of the current process
    """
    if drop_mask is not None:
        batch_drop_mask = drop_mask[:len(X)]
        X_active = X[batch_drop_mask == 0]
        y_active = y[batch_drop_mask == 0]
    else:
        X_active = X
        y_active = y

    scores = np.zeros(len(X_active))
    idx_blocks = torch.split(torch.arange(len(X_active)), block_size)

    accum_grad = None

    for idx_block in idx_blocks:
        curr_X = X_active[idx_block]
        curr_y = y_active[idx_block]

        # Compute per-block gradients with clipping
        accum_grad_block, _, last_layer_norms, _ = clip_and_accum_grads_block(
            model, curr_X, curr_y, optimizer, criterion, max_grad_norm,
            device=device, aug_mult=aug_mult, aug_fn=aug_fn
        )

        # Accumulate gradients
        if accum_grad is None:
            accum_grad = accum_grad_block
        else:
            with torch.no_grad():
                for name in accum_grad:
                    accum_grad[name] += accum_grad_block[name]

        scores[idx_block] = last_layer_norms


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
    
    return accum_grad, drop_mask



