"""
Generate gradient space attack by finding the least updated parameter direction.

This script:
1. Trains a DP-SGD model with defense enabled
2. Tracks total parameter updates during training
3. Identifies the parameter dimension with the smallest absolute update
4. Constructs a canary gradient pointing in that direction
5. Sets the norm to be slightly larger than the noise level
6. Saves the gradient to a .pt file
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
from utils.data import load_data
from utils.dpsgd import clip_and_accum_grads
from utils.dpsgd import DefenseConfig


def train_and_find_least_updated_direction(model_name, X, y, epsilon, delta, max_grad_norm,
                                         n_epochs, lr, batch_size, out_dim, device='cuda:0'):
    """
    Train a DP-SGD model with defense and find the least updated parameter direction.
    Returns the index of the least updated parameter dimension and the noise level.
    """

    # Move everything to device
    device = torch.device(device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)

    # Initialize model
    if model_name == 'lstm':
        vocab_size = out_dim
        model = Models[model_name](vocab_size=vocab_size, out_dim=out_dim).to(device)
    else:
        model = Models[model_name](X.shape, out_dim=out_dim).to(device)

    if model_name == 'cnn':
        # Xavier initialization
        def init_weights(m):
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)
        model.apply(init_weights)
    else:
        # Kaiming initialization for WideResNet
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

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

    # Store initial parameters
    initial_params = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            initial_params[name] = param.detach().clone()

    # Data loader
    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False
    )
    
    # Initialize scores array for defense mechanism
    n_samples = len(X)
    scores = np.zeros(n_samples, dtype=np.float32)

    for epoch in range(n_epochs):
        epoch_start = time.time()
        optimizer.zero_grad()
        
        # Reset scores for this epoch
        scores.fill(0.0)

        for batch_idx, (curr_X, curr_y) in enumerate(loader):
            curr_X, curr_y = curr_X.to(device), curr_y.to(device)
            batch_start_idx = batch_idx * batch_size
            batch_end_idx = min(batch_start_idx + len(curr_X), n_samples)
            
            # Create drop_mask: 0 = keep sample, no filtering
            drop_mask = np.zeros(len(curr_X), dtype=np.int32)
            
            # Global indices for this batch
            global_indices = torch.arange(batch_start_idx, batch_end_idx, device=device)

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
                drop_mask=drop_mask,
                block_size=len(curr_X),  # Process entire batch at once
                scores=scores,  # Pass the scores array to be updated
                device=device,
                global_indices=global_indices,
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

                        # Average the noisy gradient over batch size
                        grad.div_(float(batch_size))

                        if param.grad is None:
                            param.grad = grad.clone()
                        else:
                            param.grad.copy_(grad)

            optimizer.step()
            optimizer.zero_grad()

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch} completed in {epoch_time:.2f}s")

    # Compute total parameter updates
    print("Computing parameter updates...")
    total_updates = []
    param_indices = []
    global_idx = 0

    for name, param in model.named_parameters():
        if param.requires_grad and name in initial_params:
            update = param.detach() - initial_params[name]
            flat_update = update.view(-1).cpu().numpy()

            total_updates.extend(np.abs(flat_update))
            param_indices.extend(range(global_idx, global_idx + len(flat_update)))
            global_idx += len(flat_update)

    total_updates = np.array(total_updates)

    # Find the parameter dimension with the smallest absolute update
    min_update_idx = np.argmin(total_updates)
    min_update_value = total_updates[min_update_idx]

    print(f"Least updated parameter dimension: {min_update_idx}")
    print(f"Absolute update at that dimension: {min_update_value:.2e}")

    # Calculate noise level
    noise_level = 0.0
    if noise_multiplier > 0 and max_grad_norm is not None:
        noise_level = noise_multiplier * max_grad_norm
        print(f"DP noise level: {noise_level:.6f}")

    return min_update_idx, noise_level


def create_least_updated_gradient(model, target_index, target_norm, device='cuda'):
    """
    Create a 1-hot gradient vector at the specified parameter index with the given L∞ norm.
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

    if target_index < 0 or target_index >= total_elements:
        raise ValueError(f"target_index {target_index} is out of bounds for model with {total_elements} parameters")

    crafted_grad = {}

    for name, info in params.items():
        param = info['param']
        if param.requires_grad:
            grad = torch.zeros_like(param)
            if info['start_idx'] <= target_index < info['end_idx']:
                local_idx = target_index - info['start_idx']
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
    parser.add_argument('--output', type=str, default='least_updated_canary.pt', help='output .pt file path')
    parser.add_argument('--noise_margin', type=float, default=1.1, help='norm multiplier above noise level (default: 1.1)')

    args = parser.parse_args()

    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Load data
    print("Loading data...")
    X, y, out_dim = load_data(args.data_name, n_df=None)

    # Train model and find least updated direction
    print("Training model and finding least updated direction...")
    least_updated_idx, noise_level = train_and_find_least_updated_direction(
        model_name=args.model_name,
        X=X,
        y=y,
        epsilon=args.epsilon,
        delta=args.delta,
        max_grad_norm=args.max_grad_norm,
        n_epochs=args.n_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        out_dim=out_dim
    )

    # Calculate target norm (slightly above noise level)
    target_norm = args.noise_margin * noise_level
    print(f"Target norm: {target_norm:.6f} (noise level: {noise_level:.6f}, margin: {args.noise_margin})")

    # Create canary gradient with the computed parameters
    print(f"Creating canary gradient at parameter index {least_updated_idx}...")
    dummy_model = Models[args.model_name](X.shape, out_dim=out_dim)
    canary_gradient = create_least_updated_gradient(dummy_model, least_updated_idx, target_norm)

    # Save to file
    torch.save(canary_gradient, args.output)
    print(f"Saved gradient canary to {args.output}")

    # Verify the norm and position
    flat_grad = torch.cat([g.view(-1) for g in canary_gradient.values()])
    actual_norm = flat_grad.abs().max().item()
    non_zero_pos = (flat_grad.abs() > 0).nonzero(as_tuple=True)[0]
    if len(non_zero_pos) > 0:
        actual_pos = non_zero_pos[0].item()
        print(f"Actual L∞ norm: {actual_norm:.6f}")
        print(f"Non-zero position: {actual_pos} (expected: {least_updated_idx})")
    else:
        print("Warning: No non-zero elements found in gradient!")


if __name__ == '__main__':
    main()
