"""Auditing DP-SGD in black-box setting - Modified for model parallelism"""
import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import numpy as np
import argparse
from opacus.accountants.utils import get_noise_multiplier
from torch.utils.data import TensorDataset, DataLoader, Dataset
import dill

from models import Models
from models.wideresnet import WSConv2d
from utils.data import load_data
from utils.dpsgd import clip_and_accum_grads, DefenseConfig
from utils.audit import compute_eps_lower_from_mia, compute_eps_lower_from_mia_given_t
from utils.clipbkd import craft_clipbkd

from models.lstm import LSTM
from opacus.grad_sample import GradSampleModule


import torch.nn.functional as F
import torchvision.transforms.v2 as v2

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def fgsm_attack(model, X, y, epsilon=0.1, max_iter=10, alpha=0.01):
    """
    Perform iterative FGSM (I-FGSM/PGD) targeted attack to generate adversarial example.
    
    This implements a targeted attack that minimizes the cross-entropy loss for the target 
    class y, causing the model to misclassify the input as the target class. The attack 
    uses projected gradient descent with L∞ norm constraints.
    
    Algorithm:
        1. Initialize X_adv = X
        2. For i in range(max_iter):
            a. Compute loss = CrossEntropy(model(X_adv), y)
            b. Compute gradient: grad = ∇_{X_adv} loss
            c. Update: X_adv = X_adv - alpha * sign(grad)
            d. Project to L∞ ball: X_adv = clip(X + clip(X_adv - X, -ε, ε), 0, 1)
            e. If model predicts y, return success
        3. Return best adversarial example found
    
    Args:
        model (nn.Module): PyTorch model to attack (will be set to eval mode)
        X (torch.Tensor): Input tensor to perturb, shape (1, ...) for single sample
        y (torch.Tensor or int): Target class to fool the model into predicting
        epsilon (float): Maximum L∞ perturbation bound (default: 0.1)
        max_iter (int): Maximum number of attack iterations (default: 10)
        alpha (float): Step size for each iteration (default: 0.01)
    
    Returns:
        tuple: (X_adv, iters, success) where:
            - X_adv (torch.Tensor): Adversarial example (best found if attack fails)
            - iters (int): Number of iterations used
            - success (bool): True if attack succeeded, False otherwise
    
    Raises:
        AssertionError: If epsilon <= 0, alpha not in (0, epsilon], or max_iter <= 0
    
    Reference:
        Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks", 
        ICLR 2018 (PGD attack)
    """
    # Input validation
    assert epsilon > 0, f"epsilon must be positive, got {epsilon}"
    assert 0 < alpha <= epsilon, f"alpha must be in (0, epsilon], got alpha={alpha}, epsilon={epsilon}"
    assert max_iter > 0, f"max_iter must be positive, got {max_iter}"
    
    model.eval()
    X_adv = X.clone().detach().requires_grad_(True)
    best_adv = X_adv.detach().clone()
    best_confidence = -float('inf')
    
    for i in range(max_iter):
        output = model(X_adv)
        _, predicted = torch.max(output, 1)
        # Targeted attack: success when model predicts target class y
        if predicted == y:
            return X_adv.detach(), i + 1, True
            
        # Handle both scalar and tensor y
        y_idx = y.item() if y.dim() > 0 else y
        confidence = F.softmax(output, dim=1)[0, y_idx].item()
        if confidence > best_confidence:
            best_confidence = confidence
            best_adv = X_adv.detach().clone()
        
        # Targeted attack: minimize loss to increase confidence in target class
        loss = F.cross_entropy(output, y)
        model.zero_grad()
        loss.backward()
        
        data_grad = X_adv.grad.data
        sign_data_grad = data_grad.sign()
        # Move in negative gradient direction to minimize loss
        X_adv = X_adv.detach() - alpha * sign_data_grad
        delta = X_adv - X
        delta = torch.clamp(delta, -epsilon, epsilon)
        X_adv = torch.clamp(X + delta, 0, 1).detach().requires_grad_(True)
    
    # Attack failed - return best adversarial example found
    return best_adv, max_iter, False


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
    """
    params = {}
    total_elements = 0
    
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
    
    if hot_index is None:
        hot_index = total_elements // 2 if total_elements > 0 else 0
    
    if hot_index < 0 or (total_elements > 0 and hot_index >= total_elements):
        raise ValueError(f"hot_index {hot_index} is out of bounds for model with {total_elements} parameters")
    
    crafted_grad = {}
    for name, info in params.items():
        param = info['param']
        if param.requires_grad:
            grad = torch.zeros_like(param)
            
            if info['start_idx'] <= hot_index < info['end_idx']:
                local_idx = hot_index - info['start_idx']
                flat_grad = grad.view(-1)
                flat_grad[local_idx] = 10000000
                grad = flat_grad.view(info['shape'])
                
            crafted_grad[name] = grad.unsqueeze(0)
        else:
            crafted_grad[name] = torch.zeros_like(param).unsqueeze(0)
    
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


class IndexedTensorDataset(Dataset):
    """A dataset that includes the index of each sample."""
    def __init__(self, *tensors):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        
    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors) + (index,)
        
    def __len__(self):
        return self.tensors[0].size(0)


def train_model(model_name, X, y, X_target, y_target, epsilon, delta, max_grad_norm, 
               n_epochs, lr, block_size, batch_size, init_model=None, out_dim=10, aug_mult=1,
               gradient_space_audit=False, crafted_gradient=None, defense=False, defense_k: int = 5, defense_apply_ascent=True, defense_filter_every: int = 1, device='cuda:0', generator=None, dl_generator=None, rank=0, world_size=None, defense_score_norm='linf', defense_score_fn='grad_norm', loss_volatility_k: int = 5, grad_norm_percentile_k: int = 20, grad_dir_volatility_k: int = 5, grad_dir_proj_dim: int = 64, grad_dir_proj_seed: int = 0, rand_proj_var_m: int = 10, rand_proj_var_seed: int = 0, maxmin_proj_k: int = 10, maxmin_proj_seed: int = 0, grad_rank_mode: str = 'effdim', grad_rank_eps: float = 1e-12, grad_accel_proj_dim: int = 64, grad_accel_proj_seed: int = 0, grad_jerk_proj_dim: int = 64, grad_jerk_proj_seed: int = 0, dir_unique_k: int = 5, alignment_proj_k: int = 10, alignment_proj_seed: int = 0, grad_scatter_k: int = 5, num_workers: int = 4, persistent_workers: bool = True, return_defense_state: bool = False):
    """
    Train a single model on a single GPU (no DDP).
    """

    # Move everything to the specified device
    device = torch.device(device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    
    if init_model is None:
        if model_name == 'lstm':
            vocab_size = out_dim
            model = Models[model_name](vocab_size=vocab_size, out_dim=out_dim).to(device)
        else:
            model = Models[model_name](X.shape, out_dim=out_dim).to(device)
            if model_name == 'cnn':
                xavier_init_model(model)
            else:
                init_wideresnet(model)
    else:
        model = copy.deepcopy(init_model).to(device)

    if model_name == "lstm" and not isinstance(model, GradSampleModule):
        model = GradSampleModule(model)

    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # Set DP noise
    # TODO: switch accountant
    if epsilon is not None:
        sample_rate = batch_size / len(X)
        noise_multiplier = get_noise_multiplier(
            target_epsilon=epsilon,
            target_delta=delta,
            sample_rate=sample_rate,
            epochs=n_epochs,
            accountant='rdp'
        )
        if rank == 0:
            print(f"DP config: eps={epsilon}, delta={delta}, sample_rate={sample_rate:.6f}, epochs={n_epochs}, noise_multiplier={noise_multiplier}")
    else:
        noise_multiplier = 0

    block_size = min(block_size, batch_size)

    if len(X.shape) > 2:
        aug_fn = AugmentationFunction(X.shape[2], X.shape[1])
    else:
        aug_fn = None

    # Create Dataset + DataLoader (no DDP sampler)
    dataset = IndexedTensorDataset(X, y)
    scores = np.zeros(len(dataset), dtype=np.float32)
    # 0 = active, 1 = apply gradient ascent, 2 = inactive (dropped)
    # Must be integer (NOT bool) because we rely on the 3-state semantics downstream.
    drop_mask = np.zeros(len(dataset), dtype=np.int8)
    
    sampler = torch.utils.data.RandomSampler(
        dataset,
        replacement=False,
        num_samples=None,
        generator=generator
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        pin_memory=True,
        num_workers=int(num_workers),
        persistent_workers=bool(persistent_workers) if int(num_workers) > 0 else False,
        drop_last=False,
        generator=dl_generator
    )

    prev_params = None
    prev_delta_theta = None
    theta0_params = None
    prev_losses = None
    loss_hist = None
    loss_hist_pos = None
    grad_norm_hist = None
    grad_norm_hist_pos = None
    grad_dir_hist = None
    grad_dir_hist_pos = None
    grad_dir_proj = None
    rand_proj_mat = None
    maxmin_proj_mat = None
    grad_accel_hist = None
    grad_accel_hist_pos = None
    grad_accel_proj = None
    grad_jerk_hist = None
    grad_jerk_hist_pos = None
    grad_jerk_proj = None
    dir_unique_hist = None
    dir_unique_hist_pos = None
    alignment_proj_mat = None
    canary_dropped_epoch = None
    
    for epoch in range(n_epochs):
        epoch_start = time.time()
        optimizer.zero_grad()
        print(f"Epoch: {epoch} (Active samples: {int((drop_mask == 0).sum())}/{len(drop_mask)})", end='', flush=True)

        for batch_idx, (curr_X, curr_y, global_indices) in enumerate(loader):
            curr_X, curr_y = curr_X.to(device, non_blocking=True), curr_y.to(device, non_blocking=True)
            global_indices = global_indices.to(device, non_blocking=True)

            if defense_score_fn == 'loss_momentum' and prev_losses is None:
                prev_losses = np.full((len(dataset),), np.nan, dtype=np.float32)

            if defense_score_fn == 'loss_volatility' and loss_hist is None:
                k = int(loss_volatility_k)
                if k <= 0:
                    raise ValueError(f"loss_volatility_k must be > 0, got {k}")
                loss_hist = np.full((len(dataset), k), np.nan, dtype=np.float32)
                loss_hist_pos = np.zeros((len(dataset),), dtype=np.int64)

            if defense_score_fn == 'grad_norm_percentile' and grad_norm_hist is None:
                k = int(grad_norm_percentile_k)
                if k <= 0:
                    raise ValueError(f"grad_norm_percentile_k must be > 0, got {k}")
                grad_norm_hist = np.full((len(dataset), k), np.nan, dtype=np.float32)
                grad_norm_hist_pos = np.zeros((len(dataset),), dtype=np.int64)

            if defense_score_fn == 'grad_dir_volatility' and grad_dir_hist is None:
                k = int(grad_dir_volatility_k)
                if k <= 0:
                    raise ValueError(f"grad_dir_volatility_k must be > 0, got {k}")

                # Note: grad_dir_proj will be created lazily on first batch when we know the actual gradient dimensions
                grad_dir_hist = np.full((len(dataset), k, int(grad_dir_proj_dim)), np.nan, dtype=np.float32)
                grad_dir_hist_pos = np.zeros((len(dataset),), dtype=np.int64)

            if defense_score_fn == 'norm_x_dir_uniqueness' and dir_unique_hist is None:
                k = int(dir_unique_k)
                if k <= 0:
                    raise ValueError(f"dir_unique_k must be > 0, got {k}")

                # Note: grad_dir_proj will be created lazily on first batch
                dir_unique_hist = np.full((len(dataset), k, int(grad_dir_proj_dim)), np.nan, dtype=np.float32)
                dir_unique_hist_pos = np.zeros((len(dataset),), dtype=np.int64)

            if defense_score_fn == 'rand_proj_var' and rand_proj_mat is None:
                pass  # Will be created in dpsgd.py

            if defense_score_fn == 'maxmin_proj_ratio' and maxmin_proj_mat is None:
                pass  # Will be created in dpsgd.py

            if defense_score_fn == 'alignment_with_rand_proj' and alignment_proj_mat is None:
                pass  # Will be created in dpsgd.py

            if defense_score_fn == 'grad_accel' and grad_accel_hist is None:
                # Keep a 3-step history for discrete second difference.
                # Note: grad_accel_proj will be created lazily on first batch
                grad_accel_hist = np.full((len(dataset), 3, int(grad_accel_proj_dim)), np.nan, dtype=np.float32)
                grad_accel_hist_pos = np.zeros((len(dataset),), dtype=np.int64)

            if defense_score_fn == 'grad_jerk' and grad_jerk_hist is None:
                # Keep a 4-step history for discrete third difference.
                # Note: grad_jerk_proj will be created lazily on first batch
                grad_jerk_hist = np.full((len(dataset), 4, int(grad_jerk_proj_dim)), np.nan, dtype=np.float32)
                grad_jerk_hist_pos = np.zeros((len(dataset),), dtype=np.int64)

            if defense_score_fn == 'cos_update' and prev_params is None:
                prev_params = {n: p.detach().clone() for n, p in model.named_parameters()}

            if defense_score_fn == 'cos_theta0' and theta0_params is None:
                theta0_params = {n: p.detach().clone() for n, p in model.named_parameters()}

            if defense_score_fn == 'norm_x_trajectory_orth' and theta0_params is None:
                theta0_params = {n: p.detach().clone() for n, p in model.named_parameters()}

            curr_params = {n: p.detach() for n, p in model.named_parameters()}
            if prev_params is not None:
                prev_delta_theta = {n: curr_params[n] - prev_params[n] for n in prev_params.keys()}
            else:
                prev_delta_theta = None

            if theta0_params is not None:
                theta_t_minus_theta0 = {n: curr_params[n] - theta0_params[n] for n in theta0_params.keys()}
            else:
                theta_t_minus_theta0 = None

            defense_cfg = DefenseConfig(
                score_fn=defense_score_fn,
                score_norm=defense_score_norm,
                delta_theta=prev_delta_theta,
                theta_t_minus_theta0=theta_t_minus_theta0,
                grad_norm_hist=grad_norm_hist,
                grad_norm_hist_pos=grad_norm_hist_pos,
                grad_norm_percentile_k=int(grad_norm_percentile_k),
                grad_dir_hist=grad_dir_hist,
                grad_dir_hist_pos=grad_dir_hist_pos,
                grad_dir_volatility_k=int(grad_dir_volatility_k),
                grad_dir_proj=grad_dir_proj,
                rand_proj_mat=rand_proj_mat,
                rand_proj_var_m=int(rand_proj_var_m),
                maxmin_proj_mat=maxmin_proj_mat,
                maxmin_proj_k=int(maxmin_proj_k),
                grad_rank_mode=str(grad_rank_mode),
                grad_rank_eps=float(grad_rank_eps),
                grad_accel_hist=grad_accel_hist,
                grad_accel_hist_pos=grad_accel_hist_pos,
                grad_accel_proj=grad_accel_proj,
                grad_jerk_hist=grad_jerk_hist,
                grad_jerk_hist_pos=grad_jerk_hist_pos,
                alignment_proj_mat=alignment_proj_mat,
                alignment_proj_k=int(alignment_proj_k),
                grad_jerk_proj=grad_jerk_proj,
                dir_unique_hist=dir_unique_hist,
                dir_unique_hist_pos=dir_unique_hist_pos,
                dir_unique_k=int(dir_unique_k),
                grad_scatter_k=int(grad_scatter_k)
            )
            
            # Clip & accumulate gradients (no world_size/rank needed)
            curr_accumulated_gradients, scores = clip_and_accum_grads(
                model,
                curr_X, curr_y, optimizer, criterion,
                max_grad_norm, 
                drop_mask=drop_mask[global_indices.cpu().numpy()] if drop_mask is not None else None,
                block_size=block_size,
                scores=scores,
                device=device,
                global_indices=global_indices,
                aug_mult=aug_mult, 
                aug_fn=aug_fn,
                world_size=1,  # Single GPU
                rank=0,        # Single GPU
                batch_size=batch_size,
                is_gradient_space_canary=gradient_space_audit,
                crafted_gradient=crafted_gradient,
                defense_cfg=defense_cfg,
                defense_apply_ascent=defense_apply_ascent
            )
            
            # Update projection matrices if they were lazily created in dpsgd.py
            if defense_cfg.grad_dir_proj is not None:
                grad_dir_proj = defense_cfg.grad_dir_proj
            if defense_cfg.rand_proj_mat is not None:
                rand_proj_mat = defense_cfg.rand_proj_mat
            if defense_cfg.maxmin_proj_mat is not None:
                maxmin_proj_mat = defense_cfg.maxmin_proj_mat
            if defense_cfg.alignment_proj_mat is not None:
                alignment_proj_mat = defense_cfg.alignment_proj_mat
            if defense_cfg.grad_accel_proj is not None:
                grad_accel_proj = defense_cfg.grad_accel_proj
            if defense_cfg.grad_jerk_proj is not None:
                grad_jerk_proj = defense_cfg.grad_jerk_proj
            
            drop_mask[drop_mask == 1] = 2

            # Apply the accumulated gradients
            with torch.no_grad():
                for name, param in model.named_parameters():
                    
                    if name not in curr_accumulated_gradients:
                        print(f"Warning: Parameter {name} not found in accumulated gradients")
                        continue
                        
                    grad = curr_accumulated_gradients[name].to(device)
                    
                    # Add DP noise to the sum of clipped gradients (before averaging)
                    if noise_multiplier > 0 and max_grad_norm is not None:
                        noise_std = noise_multiplier * max_grad_norm
                        noise = noise_std * torch.randn_like(grad)
                        grad.add_(noise)
                    
                    # Average the noisy gradient sum
                    batch_size_in = int(curr_X.shape[0])
                    grad.div_(float(batch_size_in))
                    
                    if param.grad is None:
                        param.grad = grad.clone()
                    else:
                        param.grad.copy_(grad)
            
            optimizer.step()
            optimizer.zero_grad()

            if defense_score_fn == 'cos_update' and prev_params is not None:
                curr_params = {n: p.detach() for n, p in model.named_parameters()}
                prev_delta_theta = {n: curr_params[n] - prev_params[n] for n in prev_params.keys()}
                prev_params = {n: curr_params[n].clone() for n in prev_params.keys()}
        
        epoch_time = time.time() - epoch_start
        print(f" | Time: {epoch_time:.2f}s")
        
        # Defense operations - only apply filtering every defense_filter_every epochs
        # GLOBAL FILTERING: Filter top k scores across entire dataset (not per-class)
        if defense and (epoch % defense_filter_every == 0):
            k = int(defense_k)
            active_mask = torch.from_numpy(drop_mask == 0)
            active_indices = active_mask.nonzero(as_tuple=True)[0]
            
            if len(active_indices) > 0:
                active_scores = torch.tensor(scores[active_indices.cpu().numpy()], device=y.device)
                _, topk_indices = torch.topk(active_scores, min(k, len(active_scores)))
                
                topk_global_indices = active_indices[topk_indices]
                
                dropped_indices = topk_global_indices.cpu().numpy()
                drop_mask[dropped_indices] = 1
                
                if X.shape[0] - 1 in dropped_indices and canary_dropped_epoch is None:
                    print(f"\n[INFO] Canary (index {X.shape[0]-1}) was dropped from the training set!", drop_mask[-1])
                    canary_dropped_epoch = int(epoch)
        
            scores.fill(0)

    if return_defense_state:
        return model, drop_mask, canary_dropped_epoch
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


def compute_per_sample_losses(model, X, y, device, batch_size=256):
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

            # Handle both classification (B, C) and sequence modeling (B, T, C)
            if curr_y.ndim == 2 and logits.ndim == 3:
                b, t, c = logits.shape
                token_losses = F.cross_entropy(
                    logits.reshape(b * t, c),
                    curr_y.reshape(b * t),
                    reduction='none'
                ).reshape(b, t)
                batch_losses = token_losses.mean(dim=1)
            else:
                batch_losses = F.cross_entropy(logits, curr_y, reduction='none')

            per_sample_losses.append(batch_losses.detach().cpu())

    model.train()
    return torch.cat(per_sample_losses, dim=0).numpy()


def save_checkpoint(out_folder, outputs, losses, all_losses, train_set_accs, test_set_accs, fit_world_only, rank=0):
    """Save checkpoint - each rank saves to its own file"""
    os.makedirs(out_folder, exist_ok=True)

    # Save with rank suffix
    suffix = f'_rank{rank}' if rank > 0 else ''
    
    random_state = {
        'np': np.random.get_state(),
        'torch': torch.random.get_rng_state()
    }
    dill.dump(random_state, open(f'{out_folder}/random_state{suffix}.dill', 'wb'))

    if fit_world_only:
        np.save(f'{out_folder}/outputs_{fit_world_only}{suffix}.npy', outputs[fit_world_only])
        np.save(f'{out_folder}/losses_{fit_world_only}{suffix}.npy', losses[fit_world_only])
        if all_losses is not None:
            np.save(f'{out_folder}/all_losses_{fit_world_only}{suffix}.npy', all_losses[fit_world_only])

        if fit_world_only == 'out':
            np.save(f'{out_folder}/train_set_accs{suffix}.npy', train_set_accs)
            np.save(f'{out_folder}/test_set_accs{suffix}.npy', test_set_accs)
    else:
        np.save(f'{out_folder}/outputs_in{suffix}.npy', outputs['in'])
        np.save(f'{out_folder}/outputs_out{suffix}.npy', outputs['out'])
        np.save(f'{out_folder}/train_set_accs{suffix}.npy', train_set_accs)
        np.save(f'{out_folder}/test_set_accs{suffix}.npy', test_set_accs)
        np.save(f'{out_folder}/losses_in{suffix}.npy', losses['in'])
        np.save(f'{out_folder}/losses_out{suffix}.npy', losses['out'])
        if all_losses is not None:
            np.save(f'{out_folder}/all_losses_in{suffix}.npy', all_losses['in'])
            np.save(f'{out_folder}/all_losses_out{suffix}.npy', all_losses['out'])


def init_run_state(out_folder, fit_world_only, rank=0):
    """Initialize fresh run state and write an initial checkpoint."""
    outputs = {'out': [], 'in': []}
    losses = {'out': [], 'in': []}
    all_losses = {'in': [], 'out': []}
    train_set_accs = []
    test_set_accs = []

    os.makedirs(out_folder, exist_ok=True)
    save_checkpoint(out_folder, outputs, losses, all_losses, train_set_accs, test_set_accs, fit_world_only, rank)

    return outputs, losses, all_losses, train_set_accs, test_set_accs


def distribute_reps(n_reps, world_size):
    """Distribute model training repetitions across GPUs"""
    reps_per_gpu = [[] for _ in range(world_size)]
    for i in range(n_reps):
        gpu_id = i % world_size
        reps_per_gpu[gpu_id].append(i)
    return reps_per_gpu


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    
    # Check if running under torchrun (distributed mode)
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        dist.init_process_group(
            backend='nccl',
            init_method='env://'
        )
        
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        rank = int(os.environ.get('RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        
        if torch.cuda.is_available():
            device = torch.device(f'cuda:{local_rank}')
            torch.cuda.set_device(device)
            print(f'[Rank {rank}] Using device: {torch.cuda.get_device_name(local_rank)}')
        else:
            device = torch.device('cpu')
            print(f'[Rank {rank}] CUDA not available, using CPU')
    else:
        # Single GPU mode (no distributed training)
        local_rank = 0
        rank = 0
        world_size = 1
        
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            torch.cuda.set_device(device)
            print(f'Single GPU mode - Using device: {torch.cuda.get_device_name(0)}')
        else:
            device = torch.device('cpu')
            print(f'Single GPU mode - Using CPU')
    
    # Parse arguments
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
    parser.add_argument('--canary_pt', type=str, default=None,
                        help='Path to a .pt canary file (torch.save). If provided, overrides --target_type and uses the loaded canary/label as the target sample.')
    parser.add_argument('--blank_alpha', type=float, default=0.0, help='interpolation factor for blank target (0.0 = fully blank, 1.0 = fully label 9 image)')
    parser.add_argument('--seed', type=int, default=0, help='seed for reproducibility')
    parser.add_argument('--out', type=str, default='exp_data/', help='folder to write results to')
    parser.add_argument('--fixed_init', type=str, nargs='?', default=None, const='', help='initialize all models to the same weights (if path provided, weights loaded from path (worst-case), else fix to some randomly chosen weights)')
    parser.add_argument('--block_size', type=int, help='process samples within a batch in blocks to conserve GPU space')
    parser.add_argument('--batch_size', type=int, help='batch size for training')
    parser.add_argument('--fit_world_only', type=str, default=None, choices=['in', 'out'], help='just fit models in world and calculate losses')
    parser.add_argument('--alpha', type=float, default=0.05, help='significance level for empirical eps estimation')
    parser.add_argument('--badnets_label', type=int, default=-1, help='assign badnets poison this label')

    # Options for Debugging
    parser.add_argument('--view_badnets', action='store_true')
    parser.add_argument('--store_canary_rank', action='store_true')
    parser.add_argument('--holdout_audit', action='store_true')
    parser.add_argument('--store_all_losses', action='store_true', help='store per-sample losses for the full dataset for each trained model')
    
    # Gradient-space audit options
    parser.add_argument('--target_class', type=int, default=0,
                        help='Target class for gradient-space audit')

    # Options for Forgetting Canary Candidates
    parser.add_argument('--defense', action='store_true', help='use filtering defense during audit')
    parser.add_argument('--defense_k', type=int, default=5, help='number of samples dropped per class per epoch when defense is enabled')
    parser.add_argument('--defense_apply_ascent', action='store_true', default=True, help='apply gradient ascent to high-scoring samples (default: True when defense is enabled)')
    parser.add_argument('--defense_filter_every', type=int, default=1, help='apply defense filtering every N epochs (default: 1, i.e., every epoch)')
    parser.add_argument('--aug_mult', type=int, default=1, help='augmentation multiplier (default: 1)')
    parser.add_argument('--defense_score_norm', type=str, default='linf', choices=['linf', 'l2', 'l1'], help='norm used to score per-sample gradients for defense (linf, l2, or l1)')
    parser.add_argument('--defense_score_fn', type=str, default='grad_norm', choices=['grad_norm', 'grad_norm_percentile', 'grad_dir_volatility', 'rand_proj_var', 'maxmin_proj_ratio', 'gradient_rank', 'grad_accel', 'grad_jerk', 'norm_x_dir_uniqueness', 'alignment_with_rand_proj', 'gradient_sparsity', 'gradient_kurtosis', 'grad_dir_change_rate', 'norm_x_trajectory_orth', 'gradient_scatter', 'fisher', 'inv_confidence', 'prediction_margin', 'pred_entropy', 'cos_update', 'cos_theta0'], help='score function used for defense (grad_norm, grad_norm_percentile, grad_dir_volatility, rand_proj_var, maxmin_proj_ratio, gradient_rank, grad_accel, grad_jerk, norm_x_dir_uniqueness, alignment_with_rand_proj, gradient_sparsity, gradient_kurtosis, grad_dir_change_rate, norm_x_trajectory_orth, gradient_scatter, fisher, inv_confidence, prediction_margin, pred_entropy, cos_update, or cos_theta0)')
    parser.add_argument('--grad_norm_percentile_k', type=int, default=20, help='lookback window for grad_norm_percentile score (percentile of current grad norm within last k observed grad norms per sample)')
    parser.add_argument('--grad_dir_volatility_k', type=int, default=5, help='lookback window for grad_dir_volatility score (mean(1 - cos_sim(curr_dir, past_dir)) over last k directions)')
    parser.add_argument('--grad_dir_proj_dim', type=int, default=64, help='projection dimension for grad_dir_volatility direction embedding')
    parser.add_argument('--grad_dir_proj_seed', type=int, default=0, help='seed for grad_dir_volatility random projection')
    parser.add_argument('--dir_unique_k', type=int, default=5, help='lookback window K for norm_x_dir_uniqueness score (std of cos sims to last K directions)')

    parser.add_argument('--rand_proj_var_m', type=int, default=10, help='number of random directions for rand_proj_var score')
    parser.add_argument('--rand_proj_var_seed', type=int, default=0, help='seed for rand_proj_var random directions')
    parser.add_argument('--maxmin_proj_k', type=int, default=10, help='number of random directions for maxmin_proj_ratio score')
    parser.add_argument('--maxmin_proj_seed', type=int, default=0, help='seed for maxmin_proj_ratio random directions')
    parser.add_argument('--grad_rank_mode', type=str, default='effdim', choices=['effdim', 'entropy'], help="mode for gradient_rank score: 'effdim' uses (||g||1/||g||2)^2/d, 'entropy' uses ||g||2 * H(|g|/||g||1)")
    parser.add_argument('--grad_rank_eps', type=float, default=1e-12, help='epsilon for numerical stability in gradient_rank score')
    parser.add_argument('--grad_accel_proj_dim', type=int, default=64, help='projection dimension for grad_accel score')
    parser.add_argument('--grad_accel_proj_seed', type=int, default=0, help='seed for grad_accel random projection')
    parser.add_argument('--grad_jerk_proj_dim', type=int, default=64, help='projection dimension for grad_jerk score')
    parser.add_argument('--grad_jerk_proj_seed', type=int, default=0, help='seed for grad_jerk random projection')
    parser.add_argument('--alignment_proj_k', type=int, default=10, help='number of random directions for alignment_with_rand_proj score')
    parser.add_argument('--alignment_proj_seed', type=int, default=0, help='seed for alignment_with_rand_proj random directions')
    parser.add_argument('--grad_scatter_k', type=int, default=5, help='lookback window for gradient_scatter score (number of recent gradients to track)')

    args = parser.parse_args()
    if args.max_grad_norm == -1:
        args.max_grad_norm = None

    # Reproducibility
    np.random.seed(args.seed + rank)  # Different seed per rank
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)  # Different seed per rank for CUDA

    out_folder = f'{args.out}/{args.data_name}_{args.model_name}_eps{args.epsilon}'
    os.makedirs(out_folder, exist_ok=True)
    os.makedirs(f'{out_folder}/models', exist_ok=True)

    # Load data
    if rank == 0:
        print('Loading data')
    if args.n_df == 1:
        X_out, y_out, out_dim = load_data(args.data_name, 1)
    else:
        X_out, y_out, out_dim = load_data(args.data_name, args.n_df - 1)

    # Initialize model
    if rank == 0:
        print('Initializing model')
    init_model = None
    if args.fixed_init is not None:
        init_model = Models[args.model_name](X_out.shape, out_dim=out_dim)
        if args.model_name == 'cnn':
            xavier_init_model(init_model)
        else:
            init_wideresnet(init_model)
        if args.fixed_init == '':
            if args.model_name == 'cnn':
                xavier_init_model(init_model)
            else:
                init_wideresnet(init_model)
        else:
            init_model.load_state_dict(torch.load(args.fixed_init))
            X_out, y_out = X_out[len(X_out) // 2:], y_out[len(y_out) // 2:]

    # Craft target
    if rank == 0:
        print('Crafting target data point')

    if args.canary_pt is not None:
        if not os.path.exists(args.canary_pt):
            raise FileNotFoundError(f"--canary_pt not found: {args.canary_pt}")
        payload = torch.load(args.canary_pt, map_location='cpu')

        if isinstance(payload, dict):
            if 'canary' not in payload:
                raise KeyError(f"Canary .pt dict must contain key 'canary'. Found keys: {list(payload.keys())}")
            target_X = payload['canary']
            if 'target_label' in payload:
                target_y_val = payload['target_label']
            elif 'canary_label' in payload:
                target_y_val = payload['canary_label']
            elif 'label' in payload:
                target_y_val = payload['label']
            elif 'true_label' in payload:
                target_y_val = payload['true_label']
            elif 'audit_label' in payload:
                target_y_val = payload['audit_label']
            else:
                target_y_val = 9

        elif torch.is_tensor(payload):
            target_X = payload
            target_y_val = 9
        else:
            raise TypeError(f"Unsupported canary_pt payload type: {type(payload)}")

        if torch.is_tensor(target_y_val):
            target_y = target_y_val.clone().detach().long().view(-1)
        else:
            target_y = torch.tensor([int(target_y_val)], dtype=torch.long)

        if not torch.is_tensor(target_X):
            target_X = torch.tensor(target_X)

        target_X = target_X.clone().detach()
        if target_X.ndim == X_out.ndim - 1:
            target_X = target_X.unsqueeze(0)
        if target_X.ndim != X_out.ndim:
            raise ValueError(f"Loaded canary has shape {tuple(target_X.shape)} but expected {tuple(X_out[[0]].shape)}")

        if rank == 0:
            print(f"Loaded canary from {args.canary_pt}: X={tuple(target_X.shape)}, y={target_y.tolist()}")
    else:
        # check for data_names + target_types that don't match
        if args.data_name == 'mnist':
            pass # compatible with all canaries

        elif args.data_name == 'cifar10':
            pass # compatible with all canaries
        elif args.data_name == 'cifar100':
            pass # compatible with all canaries
        elif args.data_name == 'purchase':
            # only compatible with blank
            if args.target_type != 'blank':
                raise Exception("Canary type does not support tabular data.")
        elif args.data_name == 'tiny_shakespeare':
            if args.target_type != 'empty_sequence':
                raise Exception("For tiny_shakespeare, only target_type='empty_sequence' is supported.")
        elif args.target_type == 'empty_sequence':
            raise Exception("Target type 'empty_sequence' is only valid with data_name='tiny_shakespeare'.")

        # Craft target
        if args.target_type == 'gradient_space_canary':
            target_X = X_out[-1].unsqueeze(0)
            target_y = y_out[-1].unsqueeze(0)
            if rank == 0:
                print("Using gradient-space canary")
        elif args.target_type == 'blank':
            blank_img = torch.zeros_like(X_out[[0]])
            if args.blank_alpha > 0:
                label_9_indices = (y_out == 9).nonzero(as_tuple=True)[0]
                if len(label_9_indices) == 0:
                    raise ValueError("No label 9 samples found in dataset")
                label_9_img = X_out[label_9_indices[0]].unsqueeze(0)
                target_X = args.blank_alpha * label_9_img + (1 - args.blank_alpha) * blank_img
            else:
                target_X = blank_img
            target_y = torch.from_numpy(np.array([9]))
        elif args.target_type == 'badnets':
            target_X = X_out[-1]
            target_y = torch.tensor(args.badnets_label)
            target_X[:, -4:, -4:] = torch.max(target_X)
            target_X = target_X.unsqueeze(0)
            target_y = target_y.unsqueeze(0)
        elif args.target_type == 'sanity_check':
            target_X = X_out[-1].unsqueeze(0)
            target_y = y_out[-1].unsqueeze(0)
        elif args.target_type == 'clipbkd':
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
                aug_mult=1,  # No augmentation for FGSM
                rank=rank,
                world_size=world_size,
                gradient_space_audit=False,
                defense=False,
                defense_k=int(args.defense_k)
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
            target_X, iters_used, success = fgsm_attack(
                fgsm_model, 
                original_X, 
                target_class, 
                epsilon=0.1,  # Maximum perturbation
                max_iter=20,  # Maximum iterations
                alpha=0.01    # Step size
            )
            target_y = target_class
            
            # Validate attack results
            with torch.no_grad():
                fgsm_model.eval()
                output = fgsm_model(target_X)
                pred = output.argmax(dim=1)
                perturbation = (target_X - original_X).abs().max().item()
                
                if success:
                    print(f"FGSM attack succeeded in {iters_used} iterations")
                else:
                    print(f"FGSM attack failed after {iters_used} iterations (using best adversarial example found)")
                
                print(f"  Predicted class: {pred.item()}, Target class: {target_class.item()}")
                print(f"  Perturbation L∞ norm: {perturbation:.6f} (epsilon={0.1})")
                print(f"  Attack actually fooled model: {pred.item() == target_class.item()}")
            
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
        elif args.target_type == 'empty_sequence':
            # sequence length (same as existing chunks)
            seq_len = X_out.shape[1]
            target_X = torch.zeros((1, seq_len), dtype=torch.long)
            target_y = torch.full((1, seq_len), 9, dtype=torch.long)
        elif os.path.exists(args.target_type):
            # pre-crafted target sample
            target_X = torch.from_numpy(np.load(args.target_type))
            if init_model is not None:
                target_y =  choose_worstcase_label(init_model, target_X)
            else:
                target_y = torch.from_numpy(np.array([9]))
        else:
            raise Exception(f'Target {args.target_type} not found')

    # Define datasets
    X_in, y_in = torch.vstack((X_out[:-1], target_X)), torch.cat((y_out[:-1], target_y))
    X_test, y_test, _ = load_data(args.data_name, None, split='test')

    generator = torch.Generator().manual_seed(0)
    dl_generator = torch.Generator().manual_seed(1)
    
    if rank == 0:
        print('Training models')
    
    # Initialize run state (no resume support)
    worlds = [args.fit_world_only] if args.fit_world_only else ['in', 'out']
    
    outputs, losses, all_losses, train_set_accs, test_set_accs = init_run_state(
        out_folder, args.fit_world_only, rank)
    
    # Create crafted gradient if needed
    crafted_grad = None
    if args.target_type == 'gradient_space_canary' and args.canary_pt is None:
        if rank == 0:
            print('Creating crafted gradient')
        temp_model = Models[args.model_name](X_out.shape, out_dim=out_dim).to(device)
        if args.model_name == 'cnn':
            xavier_init_model(temp_model)
        else:
            init_wideresnet(temp_model)
        crafted_grad = craft_gradient(model=temp_model, device=device)
        del temp_model

    # Distribute repetitions across GPUs
    # TODO: fix distribute reps to account for reps completed
    reps_per_gpu = distribute_reps(args.n_reps // 2, world_size)
    my_reps = reps_per_gpu[rank]
    
    if rank == 0:
        print(f"Rep distribution: {[len(r) for r in reps_per_gpu]}")
    
    for world in worlds:
        curr_X, curr_y = (X_out, y_out) if world == 'out' else (X_in, y_in)
        
        # Each rank trains its assigned models
        for rep_idx, rep in enumerate(my_reps):
            print(f"[Rank {rank}] Training rep {rep_idx+1}/{len(my_reps)} (global rep {rep})")
            
            model = train_model(
                args.model_name, 
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
                defense_k=int(args.defense_k),
                defense_filter_every=int(args.defense_filter_every),
                aug_mult=args.aug_mult,
                gradient_space_audit=(args.target_type == 'gradient_space_canary' and args.canary_pt is None and world == 'in'),
                crafted_gradient=crafted_grad if (args.target_type == 'gradient_space_canary' and args.canary_pt is None and world == 'in') else None,
                device=device,
                generator=generator,
                dl_generator=dl_generator,
                rank=rank,
                defense_score_norm=args.defense_score_norm,
                defense_score_fn=args.defense_score_fn,
                grad_norm_percentile_k=args.grad_norm_percentile_k,
                grad_dir_volatility_k=args.grad_dir_volatility_k,
                grad_dir_proj_dim=args.grad_dir_proj_dim,
                grad_dir_proj_seed=args.grad_dir_proj_seed,
                dir_unique_k=args.dir_unique_k,
                rand_proj_var_m=args.rand_proj_var_m,
                rand_proj_var_seed=args.rand_proj_var_seed,
                maxmin_proj_k=args.maxmin_proj_k,
                maxmin_proj_seed=args.maxmin_proj_seed,
                grad_rank_mode=args.grad_rank_mode,
                grad_rank_eps=args.grad_rank_eps,
                grad_accel_proj_dim=args.grad_accel_proj_dim,
                grad_accel_proj_seed=args.grad_accel_proj_seed,
                grad_jerk_proj_dim=args.grad_jerk_proj_dim,
                grad_jerk_proj_seed=args.grad_jerk_proj_seed,
                alignment_proj_k=args.alignment_proj_k,
                alignment_proj_seed=args.alignment_proj_seed,
                grad_scatter_k=args.grad_scatter_k,
                defense_apply_ascent=args.defense_apply_ascent
            )
            
            # Compute outputs and losses
            model.eval()
            with torch.no_grad():
                target_X_device = target_X.to(device)
                target_y_device = target_y.to(device)
                
                output = model(target_X_device)
                
                if args.target_type == 'gradient_space_canary' and args.canary_pt is None and world == 'in' and crafted_grad is not None:
                    # Calculate parameter update
                    final_params = {n: p.detach().clone().to(device) for n, p in model.named_parameters()}
                    init_params = {n: p.detach().clone().to(device) for n, p in init_model.named_parameters()}
                    
                    # Calculate cosine similarity
                    update = {n: final_params[n] - init_params[n] for n, p in final_params.items()}
                    flat_crafted_grad = torch.cat([g.view(-1) for g in crafted_grad.values()])
                    flat_update = torch.cat([p.view(-1) for p in update.values()])
                    
                    flat_crafted_grad = flat_crafted_grad / (flat_crafted_grad.norm() + 1e-10)
                    flat_update = flat_update / (flat_update.norm() + 1e-10)
                    
                    cos_sim = (flat_crafted_grad * flat_update).sum().item()
                    loss = cos_sim
                else:
                    loss = -nn.CrossEntropyLoss()(output, target_y_device).cpu().item()
                
                # Store locally - no gathering needed
                outputs[world].append(output[0].cpu().numpy())
                losses[world].append(loss)

            if args.store_all_losses:
                if args.target_type == 'gradient_space_canary' and args.canary_pt is None and world == 'in':
                    # Not a per-sample loss-based audit; keep placeholder for alignment.
                    all_losses[world].append(np.array([], dtype=np.float32))
                else:
                    all_losses[world].append(
                        compute_per_sample_losses(model, curr_X, curr_y, device=device)
                    )
            
            # Each rank saves its own checkpoint
            # TODO: fix save_checkpoint to account for randomness
            save_checkpoint(out_folder, outputs, losses, all_losses, train_set_accs, test_set_accs, args.fit_world_only, rank)
            
            # Get test set accuracy from first 5 reps
            if rep < 5 and world == 'in':
                if len(X_out) > 0:
                    train_set_accs.append(test_model(model, X_in, y_in))
                    print(f'[Rank {rank}] Train set acc:', train_set_accs[-1])
                test_set_accs.append(test_model(model, X_test, y_test))
                print(f'[Rank {rank}] Test set acc:', test_set_accs[-1])
                save_checkpoint(out_folder, outputs, losses, all_losses, train_set_accs, test_set_accs, args.fit_world_only, rank)
        
        # After all reps in this world
        outputs[world] = np.array(outputs[world])

    # Synchronize only in distributed mode
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        if torch.cuda.is_available():
            dist.barrier(device_ids=[local_rank])
        else:
            dist.barrier()

    # Final audit - only rank 0 needs to combine results from all ranks
    if rank == 0:
        print("\n[Rank 0] Combining results from all ranks...")
        
        # Load results from all rank files
        combined_outputs = {'in': [], 'out': []}
        combined_losses = {'in': [], 'out': []}
        combined_all_losses = {'in': [], 'out': []}
        combined_train_accs = []
        combined_test_accs = []
        
        for r in range(world_size):
            suffix = f'_rank{r}' if r > 0 else ''
            try:
                if not args.fit_world_only:
                    combined_outputs['in'].extend(np.load(f'{out_folder}/outputs_in{suffix}.npy'))
                    combined_outputs['out'].extend(np.load(f'{out_folder}/outputs_out{suffix}.npy'))
                    combined_losses['in'].extend(np.load(f'{out_folder}/losses_in{suffix}.npy'))
                    combined_losses['out'].extend(np.load(f'{out_folder}/losses_out{suffix}.npy'))
                    if args.store_all_losses and os.path.exists(f'{out_folder}/all_losses_in{suffix}.npy'):
                        combined_all_losses['in'].extend(np.load(f'{out_folder}/all_losses_in{suffix}.npy', allow_pickle=True))
                    if args.store_all_losses and os.path.exists(f'{out_folder}/all_losses_out{suffix}.npy'):
                        combined_all_losses['out'].extend(np.load(f'{out_folder}/all_losses_out{suffix}.npy', allow_pickle=True))
                    if os.path.exists(f'{out_folder}/train_set_accs{suffix}.npy'):
                        combined_train_accs.extend(np.load(f'{out_folder}/train_set_accs{suffix}.npy'))
                    if os.path.exists(f'{out_folder}/test_set_accs{suffix}.npy'):
                        combined_test_accs.extend(np.load(f'{out_folder}/test_set_accs{suffix}.npy'))
                else:
                    combined_outputs[args.fit_world_only].extend(np.load(f'{out_folder}/outputs_{args.fit_world_only}{suffix}.npy'))
                    combined_losses[args.fit_world_only].extend(np.load(f'{out_folder}/losses_{args.fit_world_only}{suffix}.npy'))
                    if args.store_all_losses and os.path.exists(f'{out_folder}/all_losses_{args.fit_world_only}{suffix}.npy'):
                        combined_all_losses[args.fit_world_only].extend(
                            np.load(f'{out_folder}/all_losses_{args.fit_world_only}{suffix}.npy', allow_pickle=True)
                        )
            except FileNotFoundError:
                print(f"Warning: Could not find results for rank {r}")
        
        # Save combined results
        if not args.fit_world_only:
            np.save(f'{out_folder}/outputs_in.npy', combined_outputs['in'])
            np.save(f'{out_folder}/outputs_out.npy', combined_outputs['out'])
            np.save(f'{out_folder}/losses_in.npy', combined_losses['in'])
            np.save(f'{out_folder}/losses_out.npy', combined_losses['out'])
            if args.store_all_losses:
                np.save(f'{out_folder}/all_losses_in.npy', np.array(combined_all_losses['in'], dtype=object))
                np.save(f'{out_folder}/all_losses_out.npy', np.array(combined_all_losses['out'], dtype=object))
            if combined_train_accs:
                np.save(f'{out_folder}/train_set_accs.npy', combined_train_accs)
            if combined_test_accs:
                np.save(f'{out_folder}/test_set_accs.npy', combined_test_accs)
        else:
            np.save(f'{out_folder}/outputs_{args.fit_world_only}.npy', combined_outputs[args.fit_world_only])
            np.save(f'{out_folder}/losses_{args.fit_world_only}.npy', combined_losses[args.fit_world_only])
            if args.store_all_losses:
                np.save(
                    f'{out_folder}/all_losses_{args.fit_world_only}.npy',
                    np.array(combined_all_losses[args.fit_world_only], dtype=object)
                )
        
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

                # Calculate empirical epsilon using GDP
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
            
            emp_eps_loss, mia_scores, mia_labels = audit_canary(combined_losses, args)

            np.save(f'{out_folder}/emp_eps_loss.npy', [emp_eps_loss])
            np.save(f'{out_folder}/mia_scores.npy', mia_scores)
            np.save(f'{out_folder}/mia_labels.npy', mia_labels)
        
            print(f'Theoretical eps: {args.epsilon}')
            print(f'Empirical eps: {emp_eps_loss}')

        if combined_train_accs:
            print(f'Train set accuracy: {np.mean(combined_train_accs) * 100:.3f}%')
        if combined_test_accs:
            print(f'Test set accuracy: {np.mean(combined_test_accs) * 100:.3f}%')
    
    print(f"[Rank {rank}] Finished!")

    # Only destroy process group if we initialized it (distributed mode)
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        dist.destroy_process_group()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nInterrupted by user')
    except Exception as e:
        print(f'\nError in main: {str(e)}')
        import traceback
        traceback.print_exc()