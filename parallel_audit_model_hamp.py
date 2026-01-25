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
from sklearn.linear_model import LogisticRegression

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


def compute_p_from_target_entropy(gamma, num_classes, max_iter=100, tol=1e-6):
    """
    Solve for probability p given target entropy gamma.
    
    The soft label distribution is: [p, q, q, ..., q] where q = (1-p)/(C-1)
    Entropy: H = -p*log(p) - (C-1)*q*log(q)
    
    We solve for p such that H = gamma using binary search.
    
    Args:
        gamma: Target entropy as percentile of max entropy (0 to 1)
               e.g., gamma=0.95 means 95% of log(num_classes)
        num_classes: Number of classes
        max_iter: Maximum iterations for binary search
        tol: Tolerance for convergence
    
    Returns:
        p: Probability for ground truth class
    """
    import math
    
    # Maximum entropy is log(C) when uniform distribution
    max_entropy = math.log(num_classes)
    
    # Interpret gamma as percentile of max entropy (matching original HAMP)
    target_entropy = gamma * max_entropy
    
    # Clamp target_entropy to valid range
    target_entropy = max(0.0, min(target_entropy, max_entropy))
    
    # If target_entropy is max entropy, return uniform distribution
    if abs(target_entropy - max_entropy) < tol:
        return 1.0 / num_classes
    
    # If target_entropy is 0, return one-hot (p=1)
    if target_entropy < tol:
        return 1.0
    
    def compute_entropy(p):
        """Compute entropy for given p"""
        if p <= 0 or p >= 1:
            return 0.0
        q = (1 - p) / (num_classes - 1)
        if q <= 0:
            return 0.0
        H = -p * math.log(p) - (num_classes - 1) * q * math.log(q)
        return H
    
    # Binary search for p
    p_low, p_high = 1.0 / num_classes, 1.0
    
    for _ in range(max_iter):
        p_mid = (p_low + p_high) / 2.0
        H_mid = compute_entropy(p_mid)
        
        if abs(H_mid - target_entropy) < tol:
            return p_mid
        
        if H_mid < target_entropy:
            # Need higher entropy, decrease p (move toward uniform)
            p_high = p_mid
        else:
            # Need lower entropy, increase p (move toward one-hot)
            p_low = p_mid
    
    return (p_low + p_high) / 2.0


def generate_soft_labels(y, num_classes, gamma=0.95, device='cuda:0'):
    """
    Generate high-entropy soft labels for HAMP defense.
    
    Args:
        y: Hard labels (batch_size,)
        num_classes: Number of classes
        gamma: Target entropy as percentile of max entropy (0 to 1)
               e.g., gamma=0.95 means 95% of log(num_classes)
        device: Device to place tensors on
    
    Returns:
        soft_labels: Soft label distribution (batch_size, num_classes)
    """
    batch_size = y.shape[0]
    
    # Compute p from target entropy gamma
    p = compute_p_from_target_entropy(gamma, num_classes)
    other_prob = (1 - p) / (num_classes - 1)
    
    # Vectorized: initialize all to other_prob
    soft_labels = torch.full((batch_size, num_classes), other_prob, device=device)
    
    # Set ground truth labels to p using advanced indexing
    soft_labels[torch.arange(batch_size, device=device), y] = p
    
    return soft_labels


def kl_divergence_with_entropy_regularization(logits, soft_labels, alpha_entropy=1.0):
    """
    HAMP loss: KL divergence + entropy regularization.
    
    Loss = KL(soft_labels || softmax(logits)) - alpha * H(softmax(logits))
    
    The entropy term is SUBTRACTED to maximize output entropy (encourage uncertainty).
    
    Args:
        logits: Model output logits (batch_size, num_classes)
        soft_labels: Target soft label distribution (batch_size, num_classes)
        alpha_entropy: Weight for entropy regularization term (default: 1.0 to match original HAMP)
    
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
    
    # Combined loss: KL divergence - alpha * entropy (subtract to maximize entropy)
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


def generate_augmentations(x, num_augmentations=18, debug=False):
    """
    Generate augmented versions of input sample for HAMP attack.
    
    Uses horizontal flips and pixel shifts (±4 pixels on each axis) to create
    18 augmented versions as described in the HAMP paper.
    
    Args:
        x: Input sample (C, H, W)
        num_augmentations: Number of augmentations to generate (default: 18)
        debug: If True, print debug info about augmentation method
    
    Returns:
        augmented_samples: Tensor of shape (num_augmentations, C, H, W)
    """
    if debug:
        print("    [DEBUG] Using zero-padding translation (not torch.roll)")
    
    augmentations = []
    
    # Define shift values: -4, -2, 0, 2, 4 for both x and y
    shifts = [-4, -2, 0, 2, 4]
    
    # Generate augmentations: horizontal flip + shifts
    for flip in [False, True]:
        for shift_x in shifts[:3]:  # Use subset to get ~18 augmentations
            for shift_y in shifts[:3]:
                aug = x.clone()
                
                # Apply horizontal flip
                if flip:
                    aug = torch.flip(aug, dims=[2])  # Flip width dimension
                
                # Apply shift (translation) with proper background padding
                if shift_x != 0 or shift_y != 0:
                    # Use affine transformation for proper translation
                    C, H, W = aug.shape
                    
                    # Use the minimum value in the image as background (typically the background value)
                    # For normalized data, this is better than hardcoded zero
                    background_value = aug.min()
                    shifted = torch.full_like(aug, background_value)
                    
                    # Calculate source and destination slices
                    src_y_start = max(0, -shift_y)
                    src_y_end = min(H, H - shift_y)
                    dst_y_start = max(0, shift_y)
                    dst_y_end = min(H, H + shift_y)
                    
                    src_x_start = max(0, -shift_x)
                    src_x_end = min(W, W - shift_x)
                    dst_x_start = max(0, shift_x)
                    dst_x_end = min(W, W + shift_x)
                    
                    # Copy the shifted region
                    shifted[:, dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
                        aug[:, src_y_start:src_y_end, src_x_start:src_x_end]
                    
                    aug = shifted
                
                augmentations.append(aug)
                
                if len(augmentations) >= num_augmentations:
                    break
            if len(augmentations) >= num_augmentations:
                break
        if len(augmentations) >= num_augmentations:
            break
    
    return torch.stack(augmentations[:num_augmentations])


def generate_binary_correctness_vector(model, x, y, num_augmentations=18, device='cuda:0', apply_defense=False, verbose=False):
    """
    Generate binary correctness vector for augmentation-based MIA.
    
    Args:
        model: Trained model
        x: Input sample (1, C, H, W)
        y: True label (scalar or tensor)
        num_augmentations: Number of augmentations to use
        device: Device to run on
        apply_defense: If True, apply rank-preserving score replacement defense
        verbose: If True, print detailed logging
    
    Returns:
        binary_vector: Binary vector of shape (num_augmentations,) with 1 for correct, 0 for incorrect
    """
    model.eval()
    
    # Generate augmented versions
    x_aug = generate_augmentations(x.squeeze(0), num_augmentations)  # (num_aug, C, H, W)
    x_aug = x_aug.to(device)
    
    # Get true label
    if isinstance(y, torch.Tensor):
        y_true = y.item()
    else:
        y_true = int(y)
    
    # Query model on all augmentations
    binary_vector = []
    predictions = []
    
    if verbose:
        # Verify augmentation method on first call
        print("    [DEBUG] Checking augmentation quality...")
        print(f"    [DEBUG] Original image range: [{x.min():.4f}, {x.max():.4f}]")
        print(f"    [DEBUG] Augmented images range: [{x_aug.min():.4f}, {x_aug.max():.4f}]")
        
        # Check for backdoor patch presence
        orig = x.squeeze(0) if x.dim() == 4 else x
        C, H, W = orig.shape
        
        # Check bottom-right corner (common backdoor location)
        patch_region = orig[:, -3:, -3:]
        patch_max = patch_region.max().item()
        img_max = orig.max().item()
        
        if abs(patch_max - img_max) < 0.01:  # Patch likely present
            print(f"    [DEBUG] Possible backdoor patch detected in original (bottom-right max: {patch_max:.4f})")
            
            # Check if patch survives in augmentations
            aug_with_patch = 0
            for i in range(min(5, len(x_aug))):
                aug_patch = x_aug[i, :, -3:, -3:]
                if aug_patch.max().item() > img_max * 0.9:
                    aug_with_patch += 1
            print(f"    [DEBUG] Augmentations with visible patch in bottom-right: {aug_with_patch}/5")
        else:
            print(f"    [DEBUG] No obvious backdoor patch detected in original")
        
        # Check a shifted augmentation (index 1 should have shift_x=-4, shift_y=-4)
        first_aug = x_aug[1]
        
        # Check if images are identical (no shift applied) - ensure same device
        if torch.allclose(first_aug.cpu(), orig.cpu()):
            print("    [DEBUG] WARNING: Augmentation 1 is identical to original (no shift applied!)")
        else:
            print("    [DEBUG] Augmentation 1 differs from original (shift applied correctly)")
        
        # Check edge pixels - for a shifted image, some edges should be zero
        edge_sum = first_aug[:, :, 0].sum() + first_aug[:, :, -1].sum() + first_aug[:, 0, :].sum() + first_aug[:, -1, :].sum()
        print(f"    [DEBUG] Edge pixel sum: {edge_sum:.4f}")
        
        # Additional check: count how many edge pixels are exactly zero
        edge_pixels = torch.cat([
            first_aug[:, :, 0].flatten(),
            first_aug[:, :, -1].flatten(),
            first_aug[:, 0, :].flatten(),
            first_aug[:, -1, :].flatten()
        ])
        zero_edges = (edge_pixels == 0).sum().item()
        total_edges = edge_pixels.numel()
        print(f"    [DEBUG] Zero-valued edge pixels: {zero_edges}/{total_edges} ({100*zero_edges/total_edges:.1f}%)")
    
    with torch.no_grad():
        for i in range(num_augmentations):
            output = model(x_aug[i:i+1])
            
            if apply_defense:
                # Apply HAMP testing-time defense: rank-preserving score replacement
                input_shape = x_aug[i:i+1].shape[1:]  # (C, H, W)
                random_x = generate_random_samples(1, input_shape, device=device)
                random_output = model(random_x)
                output = rank_preserving_score_replacement(output, random_output)
            
            pred = torch.argmax(output, dim=1).item()
            predictions.append(pred)
            is_correct = 1 if pred == y_true else 0
            binary_vector.append(is_correct)
    
    binary_vector_np = np.array(binary_vector, dtype=np.float32)
    num_correct = int(binary_vector_np.sum())
    
    if verbose:
        # Count prediction distribution
        pred_counts = {}
        for p in predictions:
            pred_counts[p] = pred_counts.get(p, 0) + 1
        pred_dist = ', '.join([f"class {k}: {v}" for k, v in sorted(pred_counts.items())])
        
        print(f"    Target label: {y_true}")
        print(f"    Prediction distribution: {pred_dist}")
        print(f"    Augmentations classified correctly: {num_correct}/{num_augmentations} ({100*num_correct/num_augmentations:.1f}%)")
    
    return binary_vector_np


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
                # HAMP Defense: Generate soft labels and use KL divergence loss
                soft_labels = generate_soft_labels(curr_y, out_dim, gamma=hamp_gamma, device=device)
                loss = kl_divergence_with_entropy_regularization(logits, soft_labels, alpha_entropy=hamp_alpha_entropy)
            else:
                # Standard cross-entropy loss
                loss = F.cross_entropy(logits, curr_y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
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
            batch_losses = kl_loss + hamp_alpha_entropy * entropy

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


def _audit_from_scores(
    scores_in: np.ndarray,
    scores_out: np.ndarray,
    alpha: float,
    delta: float,
    holdout_audit: bool,
    seed: int = 0,
):
    """Compute empirical epsilon from canary scores"""
    if scores_in.shape[0] != scores_out.shape[0]:
        raise ValueError(f"Expected same number of in/out scores, got {scores_in.shape[0]} and {scores_out.shape[0]}")

    n = int(scores_in.shape[0])
    
    if holdout_audit:
        if n < 2:
            raise ValueError("holdout_audit requires at least 2 scores per world")
        
        np.random.seed(seed)
        indices = np.random.permutation(n)
        threshold_indices = indices[:n // 2]
        holdout_indices = indices[n // 2:]
        
        t_scores_in = scores_in[threshold_indices]
        t_scores_out = scores_out[threshold_indices]
        hold_scores_in = scores_in[holdout_indices]
        hold_scores_out = scores_out[holdout_indices]
    else:
        t_scores_in = scores_in
        t_scores_out = scores_out

    mia_scores = np.concatenate([t_scores_in, t_scores_out]).astype(np.float32)
    mia_labels = np.concatenate([
        np.ones(t_scores_in.shape[0], dtype=np.int64),
        np.zeros(t_scores_out.shape[0], dtype=np.int64),
    ])

    t, emp_eps = compute_eps_lower_from_mia(mia_scores, mia_labels, alpha, delta, 'GDP', n_procs=1)

    if holdout_audit:
        hold_mia_scores = np.concatenate([hold_scores_in, hold_scores_out]).astype(np.float32)
        hold_mia_labels = np.concatenate([
            np.ones(hold_scores_in.shape[0], dtype=np.int64),
            np.zeros(hold_scores_out.shape[0], dtype=np.int64),
        ])
        emp_eps = compute_eps_lower_from_mia_given_t(hold_mia_scores, hold_mia_labels, alpha, delta, t, 'GDP')

    return float(emp_eps), float(t), mia_scores, mia_labels


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
    parser.add_argument('--target_type', type=str, default='blank', help='sample to use as target (blank, mislabeled, backdoor_mislabeled, correctly_labeled, random_noise, gaussian_noise, uniform_noise, adversarial, sanity_check)')
    parser.add_argument('--canary_pt', type=str, default=None,
                        help='Path to a .pt canary file (torch.save). If provided, overrides --target_type and uses the loaded canary/label as the target sample.')
    parser.add_argument('--gradient_space_canary_pt', type=str, default=None,
                        help='Path to a .pt file containing a pre-crafted gradient space canary (dict of parameter gradients). Used when --target_type=gradient_space_canary.')
    parser.add_argument('--mislabeled_target_class', type=int, default=1,
                        help='Target class for mislabeled canary (default: 1). The canary will be a true class 0 sample relabeled as this class.')
    parser.add_argument('--backdoor_patch_size', type=int, default=3, help='Size of backdoor patch (default: 3x3 pixels)')
    parser.add_argument('--backdoor_patch_value', type=float, default=None, help='Value for backdoor patch pixels (default: None = max value in data range)')
    parser.add_argument('--backdoor_patch_location', type=str, default='bottom_right', choices=['top_left', 'top_right', 'bottom_left', 'bottom_right', 'center'], help='Location of backdoor patch (default: bottom_right)')
    parser.add_argument('--blank_alpha', type=float, default=0.0, help='interpolation factor for blank target (0.0 = fully blank, 1.0 = fully label 9 image)')
    parser.add_argument('--noise_scale', type=float, default=0.5, help='scale factor for noise-based canaries (default: 0.5)')
    parser.add_argument('--adversarial_epsilon', type=float, default=0.3, help='epsilon for adversarial canary generation (default: 0.3)')
    parser.add_argument('--adversarial_target_class', type=int, default=None, help='target class for adversarial canary (default: None = untargeted attack on random sample)')
    parser.add_argument('--n_canaries', type=int, default=1, help='number of identical canary copies to insert into training set (default: 1)')
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
    parser.add_argument('--hamp_gamma', type=float, default=0.95, help='HAMP soft label entropy percentile (0-1, default 0.95 = 95%% of max entropy)')
    parser.add_argument('--hamp_alpha_entropy', type=float, default=1.0, help='HAMP entropy regularization weight (default: 1.0 to match original HAMP)')

    args = parser.parse_args()
    
    # Map -1 to None for epsilon and max_grad_norm (non-private training)
    if args.epsilon == -1:
        args.epsilon = None
    if args.max_grad_norm == -1:
        args.max_grad_norm = None
    
    # Initialize model with SAME seed across all GPUs for fixed_init
    if rank == 0:
        print('Initializing model')
    init_model = None
    if args.fixed_init is not None:
        # Use same seed for all GPUs to ensure identical initialization
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        
        if rank == 0:
            print('Loading data')
        if args.n_df == 1:
            X_out, y_out, out_dim = load_data(args.data_name, 1)
        else:
            X_out, y_out, out_dim = load_data(args.data_name, args.n_df - 1)
        
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
    else:
        if rank == 0:
            print('Loading data')
        if args.n_df == 1:
            X_out, y_out, out_dim = load_data(args.data_name, 1)
        else:
            X_out, y_out, out_dim = load_data(args.data_name, args.n_df - 1)
    
    # NOW set per-rank seeds for everything else (after init_model is created)
    # This ensures data loading and other operations are still independent per GPU
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)
    
    if rank == 0:
        print(f"Dataset: {args.data_name}, Train size: {len(X_out)}, Out dim: {out_dim}")
    
    # Generate canary based on target_type
    if rank == 0:
        print(f'Crafting target data point (target_type={args.target_type})')
    
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
        # Generate canary based on target_type
        if args.target_type == 'blank':
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
        elif args.target_type == 'mislabeled':
            # Use a real sample from the dataset with an incorrect label
            # Take a sample from class 0 and label it as class 1
            class_0_indices = (y_out == 0).nonzero(as_tuple=True)[0]
            if len(class_0_indices) == 0:
                raise ValueError("No class 0 samples found in dataset")
            canary_idx = class_0_indices[0].item()
            target_X = X_out[canary_idx].unsqueeze(0)
            original_label = y_out[canary_idx].item()
            target_y = torch.from_numpy(np.array([args.mislabeled_target_class]))  # Mislabel
            if rank == 0:
                print(f"Mislabeled canary: Using sample at index {canary_idx} (original label: {original_label}) relabeled as class {args.mislabeled_target_class}")
        elif args.target_type == 'backdoor_mislabeled':
            # Use the LAST sample in dataset (which gets replaced) with a backdoor patch and mislabel it
            # The backdoor makes it visually distinct, so the model can learn the trigger -> label mapping
            canary_idx = len(X_out) - 1
            target_X = X_out[canary_idx].unsqueeze(0).clone()
            original_label = y_out[canary_idx].item()
            
            # Add backdoor patch
            C, H, W = target_X.shape[1:]
            patch_size = args.backdoor_patch_size
            
            # Determine patch value (default to max value in data for visibility)
            if args.backdoor_patch_value is not None:
                patch_value = args.backdoor_patch_value
            else:
                # Use max value in the entire dataset for maximum visibility
                patch_value = X_out.max().item()
            
            # Determine patch location
            if args.backdoor_patch_location == 'top_left':
                y_start, x_start = 0, 0
            elif args.backdoor_patch_location == 'top_right':
                y_start, x_start = 0, W - patch_size
            elif args.backdoor_patch_location == 'bottom_left':
                y_start, x_start = H - patch_size, 0
            elif args.backdoor_patch_location == 'bottom_right':
                y_start, x_start = H - patch_size, W - patch_size
            elif args.backdoor_patch_location == 'center':
                y_start, x_start = (H - patch_size) // 2, (W - patch_size) // 2
            
            # Apply patch to ALL channels
            target_X[0, :, y_start:y_start+patch_size, x_start:x_start+patch_size] = patch_value
            
            # Mislabel
            target_y = torch.from_numpy(np.array([args.mislabeled_target_class]))
            
            if rank == 0:
                print(f"Backdoor mislabeled canary: Sample {canary_idx} (original class {original_label}) + {patch_size}x{patch_size} patch (value={patch_value:.4f}) at {args.backdoor_patch_location} -> relabeled as class {args.mislabeled_target_class}")
        elif args.target_type == 'correctly_labeled':
            # Use a real sample from the dataset with its CORRECT label
            # This should be more memorable than mislabeled
            canary_idx = np.random.randint(0, len(X_out))
            target_X = X_out[canary_idx].unsqueeze(0)
            target_y = y_out[canary_idx].unsqueeze(0)
            if rank == 0:
                print(f"Correctly labeled canary: Using sample at index {canary_idx} (label: {target_y.item()})")
        elif args.target_type == 'random_noise':
            # Random noise canary: Gaussian noise with pixel values clipped to [0, 1]
            target_X = torch.randn_like(X_out[[0]]) * args.noise_scale + 0.5
            target_X = torch.clamp(target_X, 0, 1)
            target_y = torch.from_numpy(np.array([9]))  # Assign to class 9
            if rank == 0:
                print(f"Random noise canary: Gaussian noise with scale {args.noise_scale}, labeled as class 9")
        elif args.target_type == 'gaussian_noise':
            # Gaussian noise added to a real sample
            sample_idx = np.random.randint(0, len(X_out))
            base_sample = X_out[sample_idx].unsqueeze(0)
            noise = torch.randn_like(base_sample) * args.noise_scale
            target_X = torch.clamp(base_sample + noise, 0, 1)
            target_y = y_out[sample_idx].unsqueeze(0)
            if rank == 0:
                print(f"Gaussian noise canary: Real sample (idx {sample_idx}, class {target_y.item()}) + Gaussian noise (scale {args.noise_scale})")
        elif args.target_type == 'uniform_noise':
            # Uniform random noise canary
            target_X = torch.rand_like(X_out[[0]])
            target_y = torch.from_numpy(np.array([9]))  # Assign to class 9
            if rank == 0:
                print(f"Uniform noise canary: Uniform random values in [0, 1], labeled as class 9")
        elif args.target_type == 'adversarial':
            # Adversarial example using FGSM
            # First, we need to train a quick reference model to generate the adversarial example
            if rank == 0:
                print(f"Generating adversarial canary with epsilon={args.adversarial_epsilon}...")
            
            # Use a random sample as base
            sample_idx = np.random.randint(0, len(X_out))
            base_X = X_out[sample_idx].unsqueeze(0)
            base_y = y_out[sample_idx]
            
            # Train a quick reference model to generate adversarial example
            ref_model = Models[args.model_name](X_out.shape, out_dim=out_dim)
            if args.model_name == 'cnn':
                xavier_init_model(ref_model)
            else:
                init_wideresnet(ref_model)
            ref_model = ref_model.to('cuda' if torch.cuda.is_available() else 'cpu')
            ref_optimizer = optim.SGD(ref_model.parameters(), lr=0.1)
            
            # Quick training (10 epochs)
            ref_model.train()
            for _ in range(10):
                for i in range(0, len(X_out), 256):
                    batch_X = X_out[i:i+256].to(ref_model.parameters().__next__().device)
                    batch_y = y_out[i:i+256].to(ref_model.parameters().__next__().device)
                    ref_optimizer.zero_grad()
                    loss = F.cross_entropy(ref_model(batch_X), batch_y)
                    loss.backward()
                    ref_optimizer.step()
            
            # Generate adversarial example
            if args.adversarial_target_class is not None:
                target_class = args.adversarial_target_class
            else:
                # Untargeted: try to misclassify
                target_class = (base_y.item() + 1) % out_dim
            
            device_ref = next(ref_model.parameters()).device
            target_X, _, success = fgsm_attack(
                ref_model,
                base_X.to(device_ref),
                torch.tensor([target_class], device=device_ref),
                epsilon=args.adversarial_epsilon,
                max_iter=20,
                alpha=args.adversarial_epsilon / 10
            )
            target_X = target_X.cpu()
            target_y = torch.tensor([target_class])
            
            del ref_model
            torch.cuda.empty_cache()
            
            if rank == 0:
                print(f"Adversarial canary: Base sample (idx {sample_idx}, class {base_y.item()}) -> adversarial (class {target_class}, success={success})")
        elif args.target_type == 'sanity_check':
            target_X = X_out[-1].unsqueeze(0)
            target_y = y_out[-1].unsqueeze(0)
        else:
            raise ValueError(f"Unsupported target_type: {args.target_type}")
    
    # Define 'in' world dataset: all samples except last, plus the canary(ies)
    # Insert n_canaries identical copies of the canary
    if args.n_canaries > 1:
        # Create multiple copies of the canary
        canary_copies_X = target_X.repeat(args.n_canaries, 1, 1, 1) if target_X.ndim == 4 else target_X.repeat(args.n_canaries, 1)
        canary_copies_y = target_y.repeat(args.n_canaries)
        X_in = torch.vstack((X_out[:-args.n_canaries], canary_copies_X))
        y_in = torch.cat((y_out[:-args.n_canaries], canary_copies_y))
        if rank == 0:
            print(f"Inserted {args.n_canaries} identical canary copies into IN world (removed {args.n_canaries} samples from end of dataset)")
    else:
        X_in = torch.vstack((X_out[:-1], target_X))
        y_in = torch.cat((y_out[:-1], target_y))
    
    # Initialize run state (no resume support)
    worlds = [args.fit_world_only] if args.fit_world_only else ['in', 'out']
    
    outputs, losses, all_losses, train_set_accs, test_set_accs = init_run_state(args.out, args.fit_world_only, rank)
    
    # Canary scoring: store binary correctness vectors for augmentation-based MIA
    binary_vectors = {'in': [], 'out': []}
    
    # Distribute repetitions across GPUs
    reps_per_gpu = distribute_reps(args.n_reps // 2, world_size)
    my_reps = reps_per_gpu[rank]
    
    if rank == 0:
        print(f"Rep distribution: {[len(r) for r in reps_per_gpu]}")
    
    for world in worlds:
        curr_X, curr_y = (X_out, y_out) if world == 'out' else (X_in, y_in)
        
        # Each rank trains its assigned models
        for rep_idx, rep in enumerate(my_reps):
            print(f"[Rank {rank}] Training rep {rep_idx+1}/{len(my_reps)} (global rep {rep}, world={world})")
            
            # Create unique generators for each repetition
            # Use rep (global repetition number) to ensure uniqueness across all GPUs
            generator = torch.Generator().manual_seed(args.seed + rep * 2)
            dl_generator = torch.Generator().manual_seed(args.seed + rep * 2 + 1)
            
            model = train_model(
                args.model_name, curr_X, curr_y, target_X, target_y,
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
            
            # Generate binary correctness vector using augmentation-based MIA
            model.eval()
            
            # First check what the model predicts on the original canary
            with torch.no_grad():
                original_output = model(target_X.to(device))
                original_pred = torch.argmax(original_output, dim=1).item()
                target_y_val = target_y.item() if isinstance(target_y, torch.Tensor) else int(target_y)
                print(f"  Original canary prediction: {original_pred}, Training label: {target_y_val}")
            
            print(f"  Generating binary correctness vector for {world.upper()} world...")
            binary_vector = generate_binary_correctness_vector(
                model, target_X, target_y, num_augmentations=18, device=device, apply_defense=args.defense, verbose=True
            )
            binary_vectors[world].append(binary_vector)
            
            if args.defense:
                train_acc = test_model_hamp(model, curr_X, curr_y, batch_size=args.batch_size)
                
                losses_world = compute_per_sample_losses_hamp(
                    model, curr_X, curr_y, device,
                    batch_size=args.batch_size,
                    hamp_gamma=args.hamp_gamma,
                    hamp_alpha_entropy=args.hamp_alpha_entropy
                )
            else:
                train_acc = test_model(model, curr_X, curr_y, batch_size=args.batch_size)
                
                losses_world = compute_per_sample_losses(
                    model, curr_X, curr_y, device,
                    batch_size=args.batch_size
                )
            
            outputs[world].append(train_acc)
            losses[world].append(losses_world)
            train_set_accs.append(train_acc)
            
            num_correct = int(binary_vector.sum())
            print(f"  {world.upper()} - Train acc: {train_acc:.4f}, Canary augmentations correct: {num_correct}/18 ({100*num_correct/18:.1f}%)")
    
    # Save per-rank binary vectors
    suffix = f'_rank{rank}' if rank > 0 else ''
    if not args.fit_world_only:
        np.save(os.path.join(args.out, f'binary_vectors_in{suffix}.npy'), np.asarray(binary_vectors['in'], dtype=np.float32))
        np.save(os.path.join(args.out, f'binary_vectors_out{suffix}.npy'), np.asarray(binary_vectors['out'], dtype=np.float32))
    
    if rank == 0:
        save_checkpoint(args.out, outputs, losses, all_losses, train_set_accs, [], args.fit_world_only, rank)
        print(f"\nResults saved to {args.out}")
    
    # Synchronize across ranks
    if world_size > 1:
        dist.barrier()
    
    # Rank 0 combines results and trains attack model
    if rank == 0:
        print("\n[Rank 0] Combining results from all ranks...")
        combined_vectors_in = []
        combined_vectors_out = []
        
        for r in range(world_size):
            suffix = f'_rank{r}' if r > 0 else ''
            try:
                if not args.fit_world_only:
                    combined_vectors_in.extend(np.load(os.path.join(args.out, f'binary_vectors_in{suffix}.npy')))
                    combined_vectors_out.extend(np.load(os.path.join(args.out, f'binary_vectors_out{suffix}.npy')))
            except FileNotFoundError:
                print(f"Warning: Could not find binary vectors for rank {r}")
        
        # Save combined binary vectors
        if not args.fit_world_only:
            vectors_in_combined = np.asarray(combined_vectors_in, dtype=np.float32)
            vectors_out_combined = np.asarray(combined_vectors_out, dtype=np.float32)
            
            np.save(os.path.join(args.out, 'binary_vectors_in.npy'), vectors_in_combined)
            np.save(os.path.join(args.out, 'binary_vectors_out.npy'), vectors_out_combined)
            
            # Train logistic regression attack model with leave-one-out cross-validation
            print("\n[Rank 0] Training logistic regression attack model with leave-one-out CV...")
            
            num_shadow_models = len(vectors_in_combined)
            print(f"Number of shadow models: {num_shadow_models}")
            
            if num_shadow_models < 2:
                raise ValueError(f"Need at least 2 shadow models for leave-one-out, got {num_shadow_models}")
            
            # Prepare data: shape (num_models, num_augmentations)
            # Each row is a model's binary correctness vector for the canary
            all_features = np.vstack([vectors_in_combined, vectors_out_combined])  # (2*num_models, num_aug)
            all_membership = np.concatenate([
                np.ones(num_shadow_models, dtype=int),   # in-distribution
                np.zeros(num_shadow_models, dtype=int)   # out-of-distribution
            ])  # (2*num_models,)
            
            # Leave-one-out cross-validation: for each model, train on all others
            labelonly_scores_raw = np.empty(2 * num_shadow_models, dtype=np.float32)
            
            for target_model_idx in range(2 * num_shadow_models):
                # Create training set: all models except target_model_idx
                train_mask = np.ones(2 * num_shadow_models, dtype=bool)
                train_mask[target_model_idx] = False
                
                X_train = all_features[train_mask]
                y_train = all_membership[train_mask]
                
                # Train logistic regression on all other models
                attack_model = LogisticRegression(C=1.0, random_state=args.seed, max_iter=1000, solver='lbfgs')
                attack_model.fit(X_train, y_train)
                
                # Get score for target model
                X_test = all_features[target_model_idx:target_model_idx+1]
                score = attack_model.predict_proba(X_test)[0, 1]  # Probability of being in training set
                labelonly_scores_raw[target_model_idx] = score
                
                if target_model_idx % 10 == 0:
                    print(f"  Processed {target_model_idx}/{2*num_shadow_models} models")
            
            # Separate scores by membership
            mia_scores = labelonly_scores_raw
            mia_labels = all_membership.astype(np.int64)
            
            # Compute empirical epsilon using attack model scores
            emp_eps, threshold, _, _ = _audit_from_scores(
                mia_scores[mia_labels == 1],  # Scores for in-distribution
                mia_scores[mia_labels == 0],  # Scores for out-of-distribution
                float(args.alpha),
                float(args.delta),
                False,  # Already did holdout split above if needed
                seed=int(args.seed),
            )
            
            np.save(os.path.join(args.out, 'emp_eps.npy'), np.asarray(emp_eps, dtype=np.float32))
            np.save(os.path.join(args.out, 'mia_threshold.npy'), np.asarray(threshold, dtype=np.float32))
            np.save(os.path.join(args.out, 'mia_scores.npy'), mia_scores)
            np.save(os.path.join(args.out, 'mia_labels.npy'), mia_labels)
            
            print(f"\nAUDIT RESULTS")
            print(f"Theoretical epsilon: {args.epsilon}")
            print(f"Empirical epsilon: {emp_eps}")
    
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
