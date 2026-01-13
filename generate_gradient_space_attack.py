"""
Generate gradient space attack by tracking 6th largest L∞ norm for class 0 during DP-SGD training.

This script:
1. Trains a DP-SGD model end to end
2. Tracks the 6th largest L∞ per-sample gradient norm for class 0 samples each epoch
3. Finds the minimum of these norms across epochs
4. Constructs a canary gradient with norm equal to this minimum value
5. Saves the gradient to a .pt file
"""

import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import argparse

from models import Models
from models.wideresnet import WSConv2d
from utils.data import load_data
from utils.dpsgd import clip_and_accum_grads
from utils.dpsgd import DefenseConfig
from models.lstm import LSTM
from opacus.grad_sample import GradSampleModule
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as v2


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


class AugmentationFunction:
    def __init__(self, image_size=32, channels=3):
        self.base_transforms = v2.Compose([
            v2.RandomCrop(image_size, padding=4),
            v2.RandomHorizontalFlip(p=0.5),
        ])
    
    def __call__(self, x):
        return self.base_transforms(x)


def train_and_track_gradients(model_name, X, y, epsilon, delta, max_grad_norm,
                            n_epochs, lr, batch_size, out_dim, defense_k=5, defense_apply_ascent=False, 
                            defense_score_norm='linf', defense_score_fn='grad_norm', defense_filter_every=1,
                            loss_volatility_k=5, grad_norm_percentile_k=20, grad_dir_volatility_k=5,
                            grad_dir_proj_dim=64, dir_unique_k=5, rand_proj_var_m=10, maxmin_proj_k=10,
                            grad_rank_mode='effdim', grad_rank_eps=1e-12, grad_accel_proj_dim=64,
                            grad_jerk_proj_dim=64, alignment_proj_k=10, grad_scatter_k=5, block_size=None, 
                            aug_mult=1, generator=None, dl_generator=None, num_workers=4, persistent_workers=True, device='cuda:0'):
    """
    Train a DP-SGD model with defense enabled and track the 6th largest L∞ gradient norm for class 0 samples each epoch.
    Returns the minimum of these norms across all epochs.
    
    Args:
        defense_k: Number of samples to filter per class per epoch (default: 5)
    """

    # Move everything to device
    device = torch.device(device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    
    # Create model
    if model_name == 'lstm':
        vocab_size = out_dim
        model = Models[model_name](vocab_size=vocab_size, out_dim=out_dim).to(device)
    else:
        model = Models[model_name](X.shape, out_dim=out_dim).to(device)
        if model_name == 'cnn':
            xavier_init_model(model)
        else:
            init_wideresnet(model)

    if model_name == "lstm" and not isinstance(model, GradSampleModule):
        model = GradSampleModule(model)

    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # Set DP noise
    if epsilon is not None:
        sample_rate = batch_size / len(X)
        from opacus.accountants.utils import get_noise_multiplier
        noise_multiplier = get_noise_multiplier(
            target_epsilon=epsilon,
            target_delta=delta,
            sample_rate=sample_rate,
            epochs=n_epochs,
            accountant='rdp'
        )
        print(f"DP config: eps={epsilon}, delta={delta}, sample_rate={sample_rate:.6f}, epochs={n_epochs}, noise_multiplier={noise_multiplier}")
    else:
        noise_multiplier = 0

    block_size = min(block_size, batch_size) if block_size is not None else batch_size

    if len(X.shape) > 2:
        aug_fn = AugmentationFunction(X.shape[2], X.shape[1])
    else:
        aug_fn = None

    # Create Dataset + DataLoader (no DDP sampler)
    dataset = IndexedTensorDataset(X, y)
    scores = np.zeros(len(dataset), dtype=np.float32)
    # 0 = active, 1 = apply gradient ascent, 2 = inactive (dropped)
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

    # Track the 6th largest L∞ norm for class 0 samples each epoch
    epoch_6th_largest_norms = []

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

    for epoch in range(n_epochs):
        epoch_start = time.time()
        optimizer.zero_grad()
        print(f"Epoch: {epoch} (Active samples: {int((drop_mask == 0).sum())}/{len(drop_mask)})", end='', flush=True)

        # Track L∞ norms for class 0 samples in this epoch (will be extracted from defense scores)
        class_0_norms = []

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
                is_gradient_space_canary=False,
                crafted_gradient=None,
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

            # Extract L∞ gradient norms from defense scores (these are per-sample norms)
            # scores contains the gradient norms computed by the defense mechanism
            for local_idx in range(len(curr_X)):
                global_idx = global_indices[local_idx].item()
                if global_idx < len(scores):
                    norm_value = float(scores[global_idx])
                    # Only track class 0 samples that haven't been filtered out
                    # Exclude the last sample (index len(X)-1) as it will be replaced by the canary during audit
                    if curr_y[local_idx].item() == 0 and drop_mask[global_idx] == 0 and global_idx != len(X) - 1:
                        class_0_norms.append(norm_value)

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
        
        # Find the 6th largest L∞ norm for this epoch (if we have at least 6 samples)
        if len(class_0_norms) >= 6:
            sorted_norms = sorted(class_0_norms, reverse=True)
            sixth_largest = sorted_norms[5]  # 0-indexed, so index 5 is the 6th largest
            epoch_6th_largest_norms.append(sixth_largest)
            print(f"Epoch {epoch}: 6th largest L∞ norm for class 0 = {sixth_largest:.6f}")
        else:
            print(f"Epoch {epoch}: Only {len(class_0_norms)} class 0 samples, skipping norm tracking")

        epoch_time = time.time() - epoch_start
        print(f" | Time: {epoch_time:.2f}s")
        
        # Defense operations - only apply filtering every defense_filter_every epochs
        if defense_k > 0 and (epoch % defense_filter_every == 0):
            k = int(defense_k)
            unique_classes = torch.unique(y).cpu()
            active_mask = torch.from_numpy(drop_mask == 0)
            
            for cls in unique_classes:
                cls_indices = ((y.cpu() == cls.item()) & active_mask).nonzero(as_tuple=True)[0]
                if len(cls_indices) == 0:
                    continue
                    
                cls_scores = torch.tensor(scores[cls_indices.cpu().numpy()], device=y.device)
                _, topk_indices = torch.topk(cls_scores, min(k, len(cls_scores)))
                
                topk_global_indices = cls_indices[topk_indices]
                
                dropped_indices = topk_global_indices.cpu().numpy()
                drop_mask[dropped_indices] = 1
                
                if X.shape[0] - 1 in dropped_indices:
                    print(f"\n[INFO] Canary (index {X.shape[0]-1}) was dropped from the training set!", drop_mask[-1])
        
            scores.fill(0)

    # Find the minimum of the 6th largest norms across epochs
    if epoch_6th_largest_norms:
        min_norm = min(epoch_6th_largest_norms)
        print(f"\nMinimum of 6th largest norms across epochs: {min_norm:.6f}")
        return min_norm
    else:
        print("\nNo valid norm measurements found")
        return None


def create_canary_gradient(model, target_norm, device='cuda'):
    """
    Create a 1-hot gradient vector with the specified L∞ norm.
    Similar to craft_gradient but with custom norm.
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

    if total_elements == 0:
        raise ValueError("Model has no trainable parameters")

    # Create 1-hot gradient at the middle index
    hot_index = total_elements // 2
    crafted_grad = {}

    for name, info in params.items():
        param = info['param']
        if param.requires_grad:
            grad = torch.zeros_like(param)
            if info['start_idx'] <= hot_index < info['end_idx']:
                local_idx = hot_index - info['start_idx']
                flat_grad = grad.view(-1)
                flat_grad[local_idx] = target_norm  # Set to target norm
                grad = flat_grad.view(info['shape'])
            crafted_grad[name] = grad.unsqueeze(0)  # Add batch dimension
        else:
            crafted_grad[name] = torch.zeros_like(param).unsqueeze(0)

    return crafted_grad


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default='mnist', help='dataset to use')
    parser.add_argument('--model_name', type=str, default='lr', choices=list(Models.keys()), help='model to audit')
    parser.add_argument('--n_epochs', type=int, default=10, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--max_grad_norm', type=float, default=1, help='gradient clipping norm')
    parser.add_argument('--epsilon', type=float, default=None, help='privacy parameter epsilon')
    parser.add_argument('--delta', type=float, default=1e-5, help='privacy parameter delta')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--output', type=str, default='gradient_canary.pt', help='output .pt file path')
    parser.add_argument('--defense_k', type=int, default=5, help='number of samples to filter per class per epoch (default: 5)')
    parser.add_argument('--defense_apply_ascent', action='store_true', default=False, help='apply gradient ascent to high-scoring samples')
    parser.add_argument('--block_size', type=int, help='process samples within a batch in blocks to conserve GPU space')

    args = parser.parse_args()

    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Load data
    print("Loading data...")
    X, y, out_dim = load_data(args.data_name, n_df=None)

    # Train model and track gradients
    print("Training model and tracking gradients...")
    print(f"Defense enabled with k={args.defense_k}")
    
    # Create generators for reproducibility
    generator = torch.Generator().manual_seed(args.seed)
    dl_generator = torch.Generator().manual_seed(args.seed + 1)
    
    min_norm = train_and_track_gradients(
        model_name=args.model_name,
        X=X,
        y=y,
        epsilon=args.epsilon,
        delta=args.delta,
        max_grad_norm=args.max_grad_norm,
        n_epochs=args.n_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        out_dim=out_dim,
        defense_k=args.defense_k,
        defense_apply_ascent=args.defense_apply_ascent,
        block_size=args.block_size,
        generator=generator,
        dl_generator=dl_generator
    )

    if min_norm is None:
        print("Failed to compute gradient norm - exiting")
        return

    # Create canary gradient with the computed norm
    print(f"Creating canary gradient with norm {min_norm:.6f}...")

    # Create a dummy model to get parameter structure
    if args.model_name == 'lstm':
        dummy_model = Models[args.model_name](vocab_size=out_dim, out_dim=out_dim)
    else:
        dummy_model = Models[args.model_name](X.shape, out_dim=out_dim)

    canary_gradient = create_canary_gradient(dummy_model, min_norm)

    # Save to file in dictionary format to match other canary formats
    canary_dict = {
        'gradient': canary_gradient,
        'target_class': 0  # Default target class for gradient space canary
    }
    torch.save(canary_dict, args.output)
    print(f"Saved gradient canary to {args.output}")
    
    # Verify the norm
    flat_grad = torch.cat([g.view(-1) for g in canary_gradient.values()])
    actual_norm = flat_grad.abs().max().item()
    print(f"Actual L∞ norm of saved gradient: {actual_norm:.6f}")


if __name__ == '__main__':
    main()
