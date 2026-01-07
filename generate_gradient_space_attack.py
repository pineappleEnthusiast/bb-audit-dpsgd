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
import numpy as np
import argparse

from models import Models
from models.wideresnet import WSConv2d
from utils.data import load_data
from utils.dpsgd import clip_and_accum_grads
from utils.dpsgd import DefenseConfig
import torch.nn.functional as F


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



def train_and_track_gradients(model_name, X, y, epsilon, delta, max_grad_norm,
                            n_epochs, lr, batch_size, out_dim, device='cuda:0',
                            fixed_init=False, seed=0):
    """
    Train a DP-SGD model with defense enabled and track the 6th largest L∞ gradient norm for class 0 samples each epoch.
    Returns the minimum of these norms across all epochs.
    """

    # Move everything to device
    device = torch.device(device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)

    # Initialize model
    if fixed_init:
        torch.manual_seed(seed)
        np.random.seed(seed)
        
    if model_name == 'lstm':
        vocab_size = out_dim
        model = Models[model_name](vocab_size=vocab_size, out_dim=out_dim).to(device)
    else:
        model = Models[model_name](X.shape, out_dim=out_dim).to(device)

    if model_name == 'cnn':
        xavier_init_model(model)
    else:
        # Kaiming initialization for WideResNet
        init_wideresnet(model)


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

    # Data loader
    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False
    )

    # Track the 6th largest L∞ norm for class 0 samples each epoch
    epoch_6th_largest_norms = []

    for epoch in range(n_epochs):
        epoch_start = time.time()
        optimizer.zero_grad()

        # Track L∞ norms for class 0 samples in this epoch (will be extracted from defense scores)
        class_0_norms = []

        for batch_idx, (curr_X, curr_y) in enumerate(loader):
            curr_X, curr_y = curr_X.to(device), curr_y.to(device)

            defense_cfg = DefenseConfig(
                score_fn='grad_norm',
                score_norm='linf',
                delta_theta=None,
                theta_t_minus_theta0=None,
                grad_norm_hist=None,
                grad_norm_hist_pos=None,
                grad_norm_percentile_k=20,
                grad_dir_hist=None,
                grad_dir_hist_pos=None,
                grad_dir_volatility_k=5,
                grad_dir_proj=None,
                rand_proj_mat=None,
                rand_proj_var_m=10,
                maxmin_proj_mat=None,
                maxmin_proj_k=10,
                grad_rank_mode='effdim',
                grad_rank_eps=1e-12,
                grad_accel_hist=None,
                grad_accel_hist_pos=None,
                grad_accel_proj=None,
                grad_jerk_hist=None,
                grad_jerk_hist_pos=None,
                alignment_proj_mat=None,
                alignment_proj_k=10,
                grad_jerk_proj=None,
                dir_unique_hist=None,
                dir_unique_hist_pos=None,
                dir_unique_k=5,
                grad_scatter_k=5
            )

            # Compute accumulated gradients with DP-SGD and defense enabled
            curr_accumulated_gradients, scores = clip_and_accum_grads(
                model,
                curr_X, curr_y, optimizer, criterion,
                max_grad_norm,
                drop_mask=np.zeros(len(curr_X), dtype=np.int8),
                block_size=1,
                scores=np.zeros(len(curr_X), dtype=np.float32),  # Will be updated by defense
                device=device,
                global_indices=torch.arange(len(curr_X), device=device),
                aug_mult=1,
                aug_fn=None,
                world_size=1,
                rank=0,
                batch_size=len(curr_X),
                is_gradient_space_canary=False,
                crafted_gradient=None,
                defense_cfg=defense_cfg,
                defense_apply_ascent=False
            )

            # Extract L∞ gradient norms from defense scores (these are per-sample norms)
            # scores contains the gradient norms computed by the defense mechanism
            for sample_idx in range(len(curr_X)):
                if sample_idx < len(scores):
                    norm_value = scores[sample_idx]
                    # Only track class 0 samples
                    if curr_y[sample_idx] == 0:
                        class_0_norms.append(norm_value)

            # Apply the accumulated gradients
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in curr_accumulated_gradients:
                        grad = curr_accumulated_gradients[name].to(device)

                        # Add DP noise
                        if noise_multiplier > 0 and max_grad_norm is not None:
                            noise_std = noise_multiplier * max_grad_norm
                            noise = noise_std * torch.randn_like(grad)
                            grad.add_(noise)

                        # Average the noisy gradient
                        grad.div_(float(len(curr_X)))

                        if param.grad is None:
                            param.grad = grad.clone()
                        else:
                            param.grad.copy_(grad)

            optimizer.step()
            optimizer.zero_grad()

        # Find the 6th largest L∞ norm for this epoch (if we have at least 6 samples)
        if len(class_0_norms) >= 6:
            sorted_norms = sorted(class_0_norms, reverse=True)
            sixth_largest = sorted_norms[5]  # 0-indexed, so index 5 is the 6th largest
            epoch_6th_largest_norms.append(sixth_largest)
            print(f"Epoch {epoch}: 6th largest L∞ norm for class 0 = {sixth_largest:.6f}")
        else:
            print(f"Epoch {epoch}: Only {len(class_0_norms)} class 0 samples, skipping norm tracking")

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch} completed in {epoch_time:.2f}s")

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
    parser.add_argument('--fixed_init', action='store_true', help='Use fixed initialization for reproducibility')


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
        fixed_init=args.fixed_init,
        seed=args.seed
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

    # Save to file
    torch.save(canary_gradient, args.output)
    print(f"Saved gradient canary to {args.output}")

    # Verify the norm
    flat_grad = torch.cat([g.view(-1) for g in canary_gradient.values()])
    actual_norm = flat_grad.abs().max().item()
    print(f"Actual L∞ norm of saved gradient: {actual_norm:.6f}")


if __name__ == '__main__':
    main()
