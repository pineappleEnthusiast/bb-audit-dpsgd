"""Auditing DP-SGD in black-box setting - Modified for model parallelism with HAMP defense"""
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
from utils.audit import compute_eps_lower_from_mia, compute_eps_lower_from_mia_given_t

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
        
        # Handle both scalar and tensor y
        y_idx = y.item() if isinstance(y, torch.Tensor) else int(y)
        predicted_idx = predicted.item() if isinstance(predicted, torch.Tensor) else int(predicted)
        
        # Targeted attack: success when model predicts target class y
        if predicted_idx == y_idx:
            return X_adv.detach(), i + 1, True
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


def generate_soft_labels(y, num_classes, gamma=0.5, device='cuda:0'):
    """
    Generate high-entropy soft labels for HAMP defense.
    
    Args:
        y: Hard labels (batch_size,)
        num_classes: Number of classes
        gamma: Target entropy parameter (controls confidence level)
        device: Device to place tensors on
    
    Returns:
        soft_labels: Soft label distribution (batch_size, num_classes)
    """
    batch_size = y.shape[0]
    soft_labels = torch.zeros(batch_size, num_classes, device=device)
    
    # For each sample, assign probability p to ground truth and (1-p)/C-1 to others
    # We choose p such that entropy meets target
    # Target entropy: H = -p*log(p) - (1-p)*log((1-p)/(C-1))
    # For simplicity, we use p = gamma (e.g., 0.6 for 60% confidence on ground truth)
    
    p = gamma
    other_prob = (1 - p) / (num_classes - 1)
    
    for i in range(batch_size):
        soft_labels[i, :] = other_prob
        soft_labels[i, y[i]] = p
    
    return soft_labels


def kl_divergence_with_entropy_regularization(logits, soft_labels, alpha_entropy=0.1):
    """
    HAMP loss: KL divergence + entropy regularization.
    
    Loss = KL(soft_labels || softmax(logits)) - alpha * H(softmax(logits))
    
    Args:
        logits: Model output logits (batch_size, num_classes)
        soft_labels: Target soft label distribution (batch_size, num_classes)
        alpha_entropy: Weight for entropy regularization term
    
    Returns:
        loss: Scalar loss value
    """
    # Compute softmax probabilities
    probs = F.softmax(logits, dim=1)
    
    # KL divergence: sum(soft_labels * (log(soft_labels) - log(probs)))
    # Use log_softmax for numerical stability
    log_probs = F.log_softmax(logits, dim=1)
    kl_loss = torch.sum(soft_labels * (torch.log(soft_labels + 1e-10) - log_probs), dim=1).mean()
    
    # Entropy regularization: -sum(probs * log(probs))
    entropy = -torch.sum(probs * log_probs, dim=1).mean()
    
    # Combined loss: penalize low entropy (high confidence)
    loss = kl_loss - alpha_entropy * entropy
    
    return loss


def generate_random_samples(batch_size, input_shape, device='cuda:0'):
    """
    Generate random samples for testing-time defense.
    
    Args:
        batch_size: Number of random samples to generate
        input_shape: Shape of input (e.g., (3, 32, 32) for CIFAR-10)
        device: Device to place tensors on
    
    Returns:
        random_samples: Tensor of random samples (batch_size, *input_shape)
    """
    return torch.rand(batch_size, *input_shape, device=device)


def rank_preserving_score_replacement(original_scores, random_scores):
    """
    Replace original scores with random scores while preserving rank ordering.
    
    Args:
        original_scores: Original model output scores (batch_size, num_classes)
        random_scores: Scores from random samples (batch_size, num_classes)
    
    Returns:
        modified_scores: Modified scores with preserved ranking (batch_size, num_classes)
    """
    batch_size, num_classes = original_scores.shape
    modified_scores = torch.zeros_like(original_scores)
    
    for i in range(batch_size):
        # Get ranking of original scores
        _, original_ranking = torch.sort(original_scores[i], descending=True)
        
        # Get sorted random scores
        sorted_random, _ = torch.sort(random_scores[i], descending=True)
        
        # Assign sorted random scores to maintain original ranking
        modified_scores[i, original_ranking] = sorted_random
    
    return modified_scores


def train_model(model_name, X, y, X_target, y_target, epsilon, delta, max_grad_norm, 
               n_epochs, lr, block_size, batch_size, init_model=None, out_dim=10, aug_mult=1,
               gradient_space_audit=False, crafted_gradient=None, defense=False, device='cuda:0', generator=None, dl_generator=None, rank=0, world_size=None, num_workers: int = 4, persistent_workers: bool = True, hamp_gamma: float = 0.6, hamp_alpha_entropy: float = 0.1):
    """
    Train a single model on a single GPU (no DDP).
    When defense=True, applies HAMP soft label training instead of standard cross-entropy.
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
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # Set DP noise
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

    # Create Dataset + DataLoader
    dataset = IndexedTensorDataset(X, y)
    
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
    
    for epoch in range(n_epochs):
        epoch_start = time.time()
        optimizer.zero_grad()
        print(f"Epoch: {epoch}", end='', flush=True)

        for batch_idx, (curr_X, curr_y, global_indices) in enumerate(loader):
            curr_X, curr_y = curr_X.to(device, non_blocking=True), curr_y.to(device, non_blocking=True)
            
            # Forward pass
            model.train()
            logits = model(curr_X)
            
            if defense:
                # HAMP Training-time Defense: Generate soft labels and use KL divergence loss
                soft_labels = generate_soft_labels(curr_y, out_dim, gamma=hamp_gamma, device=device)
                loss = kl_divergence_with_entropy_regularization(logits, soft_labels, alpha_entropy=hamp_alpha_entropy)
            else:
                # Standard cross-entropy loss
                loss = F.cross_entropy(logits, curr_y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Clip gradients
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            optimizer.step()
            optimizer.zero_grad()
        
        epoch_time = time.time() - epoch_start
        print(f" | Time: {epoch_time:.2f}s")

    return model


def test_model(model, X, y, batch_size=128):
    # Get device from model parameters
    device = next(model.parameters()).device
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


def test_model_hamp(model, X, y, batch_size=128):
    """
    Test model with HAMP testing-time defense: rank-preserving score replacement.
    """
    # Get device from model parameters
    device = next(model.parameters()).device
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
            
            # Get original predictions
            original_logits = model(curr_X)
            
            # Generate random samples and get their predictions
            random_X = generate_random_samples(curr_X.shape[0], curr_X.shape[1:], device=device)
            random_logits = model(random_X)
            
            # Apply rank-preserving score replacement
            modified_logits = rank_preserving_score_replacement(original_logits, random_logits)
            
            # Get predictions from modified scores
            curr_y_hat = torch.argmax(modified_logits, dim=1)
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


def compute_per_sample_losses_hamp(model, X, y, device, batch_size=256, hamp_gamma=0.6, hamp_alpha_entropy=0.1):
    """
    Compute per-sample losses using HAMP loss function.
    """
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
            
            # Generate soft labels
            num_classes = logits.shape[1]
            soft_labels = generate_soft_labels(curr_y, num_classes, gamma=hamp_gamma, device=device)
            
            # Compute HAMP loss per sample
            probs = F.softmax(logits, dim=1)
            log_probs = F.log_softmax(logits, dim=1)
            
            # KL divergence per sample
            kl_loss = torch.sum(soft_labels * (torch.log(soft_labels + 1e-10) - log_probs), dim=1)
            
            # Entropy per sample
            entropy = -torch.sum(probs * log_probs, dim=1)
            
            # HAMP loss per sample
            batch_losses = kl_loss - hamp_alpha_entropy * entropy

            per_sample_losses.append(batch_losses.detach().cpu())

    model.train()
    return torch.cat(per_sample_losses, dim=0).numpy()


def save_checkpoint(out_folder, outputs, losses, all_losses, train_set_accs, test_set_accs, fit_world_only, rank=0):
    """Save checkpoint - each rank saves to its own file"""
    os.makedirs(out_folder, exist_ok=True)

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
    parser.add_argument('--target_type', type=str, default='blank', help='sample to use as target (blank, mislabeled, clipbkd, badnets, or path to target sample)')
    parser.add_argument('--canary_pt', type=str, default=None,
                        help='Path to a .pt canary file (torch.save). If provided, overrides --target_type and uses the loaded canary/label as the target sample.')
    parser.add_argument('--gradient_space_canary_pt', type=str, default=None,
                        help='Path to a .pt file containing a pre-crafted gradient space canary (dict of parameter gradients). Used when --target_type=gradient_space_canary.')
    parser.add_argument('--mislabeled_target_class', type=int, default=1,
                        help='Target class for mislabeled canary (default: 1). The canary will be a true class 0 sample relabeled as this class.')
    parser.add_argument('--blank_alpha', type=float, default=0.0, help='interpolation factor for blank target (0.0 = fully blank, 1.0 = fully label 9 image)')
    parser.add_argument('--seed', type=int, default=0, help='seed for reproducibility')
    parser.add_argument('--out', type=str, default='exp_data/', help='folder to write results to')
    parser.add_argument('--fixed_init', type=str, nargs='?', default=None, const='', help='initialize all models to the same weights (if path provided, weights loaded from path (worst-case), else fix to some randomly chosen weights)')
    parser.add_argument('--block_size', type=int, help='process samples within a batch in blocks to conserve GPU space')
    parser.add_argument('--batch_size', type=int, help='batch size for training')
    parser.add_argument('--fit_world_only', type=str, default=None, choices=['in', 'out'], help='just fit models in world and calculate losses')
    parser.add_argument('--alpha', type=float, default=0.05, help='significance level for empirical eps estimation')
    parser.add_argument('--badnets_label', type=int, default=-1, help='assign badnets poison this label')

    parser.add_argument('--view_badnets', action='store_true')
    parser.add_argument('--store_canary_rank', action='store_true')
    parser.add_argument('--holdout_audit', action='store_true')
    parser.add_argument('--store_all_losses', action='store_true', help='store per-sample losses for the full dataset for each trained model')
    
    parser.add_argument('--target_class', type=int, default=0,
                        help='Target class for gradient-space audit')

    parser.add_argument('--defense', action='store_true', help='use HAMP defense during training')
    parser.add_argument('--hamp_gamma', type=float, default=0.6, help='HAMP soft label confidence (gamma parameter)')
    parser.add_argument('--hamp_alpha_entropy', type=float, default=0.1, help='HAMP entropy regularization weight')

    args = parser.parse_args()
    
    # Map -1 to None for epsilon and max_grad_norm (non-private training)
    if args.epsilon == -1:
        args.epsilon = None
    if args.max_grad_norm == -1:
        args.max_grad_norm = None
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    if rank == 0:
        print('Loading data')
    if args.n_df == 1:
        X, y, out_dim = load_data(args.data_name, 1)
    else:
        X, y, out_dim = load_data(args.data_name, args.n_df - 1)
    
    if rank == 0:
        print(f"Dataset: {args.data_name}, Train size: {len(X)}, Out dim: {out_dim}")
    
    outputs, losses, all_losses, train_set_accs, test_set_accs = init_run_state(args.out, args.fit_world_only, rank)
    
    reps_per_gpu = distribute_reps(args.n_reps, world_size)
    
    for rep_idx in reps_per_gpu[rank]:
        if rank == 0:
            print(f"\n[Rep {rep_idx}] Training models...")
        
        generator = torch.Generator()
        generator.manual_seed(args.seed + rep_idx)
        dl_generator = torch.Generator()
        dl_generator.manual_seed(args.seed + rep_idx)
        
        init_model = None
        if args.fixed_init is not None:
            if args.fixed_init == '':
                init_model = Models[args.model_name](X.shape, out_dim=out_dim)
                if args.model_name == 'cnn':
                    xavier_init_model(init_model)
                else:
                    init_wideresnet(init_model)
            else:
                init_model = torch.load(args.fixed_init)
        
        for world in ['out', 'in']:
            if args.fit_world_only is not None and world != args.fit_world_only:
                continue
            
            if world == 'out':
                X_world, y_world = X, y
                X_target, y_target = torch.zeros(1, *X.shape[1:]), torch.tensor([0])
            else:
                X_world, y_world = X[:-1], y[:-1]
                X_target, y_target = X[-1:], y[-1:]
            
            model = train_model(
                args.model_name, X_world, y_world, X_target, y_target,
                args.epsilon, args.delta, args.max_grad_norm,
                args.n_epochs, args.lr, args.block_size, args.batch_size,
                init_model=init_model, out_dim=out_dim,
                defense=args.defense,
                device=str(device),
                generator=generator,
                dl_generator=dl_generator,
                rank=rank,
                hamp_gamma=args.hamp_gamma,
                hamp_alpha_entropy=args.hamp_alpha_entropy
            )
            
            if args.defense:
                train_acc = test_model_hamp(model, X_world, y_world, batch_size=args.batch_size)
                
                losses_world = compute_per_sample_losses_hamp(
                    model, X_world, y_world, device,
                    batch_size=args.batch_size,
                    hamp_gamma=args.hamp_gamma,
                    hamp_alpha_entropy=args.hamp_alpha_entropy
                )
            else:
                train_acc = test_model(model, X_world, y_world, batch_size=args.batch_size)
                
                losses_world = compute_per_sample_losses(
                    model, X_world, y_world, device,
                    batch_size=args.batch_size
                )
            
            outputs[world].append(train_acc)
            losses[world].append(losses_world)
            train_set_accs.append(train_acc)
            
            if rank == 0:
                print(f"  {world.upper()} - Train acc: {train_acc:.4f}")
    
    if rank == 0:
        save_checkpoint(args.out, outputs, losses, all_losses, train_set_accs, [], args.fit_world_only, rank)
        print(f"\nResults saved to {args.out}")
    
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
