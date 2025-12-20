"""Auditing DP-SGD in black-box setting using Opacus"""
import os
import sys
import time
import copy
import datetime
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Dataset
from tqdm import tqdm
import numpy as np
import argparse
import dill

# Suppress Opacus warning about new dataset objects (expected when using defense)
warnings.filterwarnings("ignore", message="PrivacyEngine detected new dataset object")

from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from opacus.accountants.utils import get_noise_multiplier
from opacus.utils.batch_memory_manager import BatchMemoryManager

import matplotlib.pyplot as plt

from models import Models
from models.wideresnet import WSConv2d
from utils.data import load_data
from utils.audit import compute_eps_lower_from_mia, compute_eps_lower_from_mia_given_t, compute_eps_lower_single
from sklearn.linear_model import LogisticRegression
from privacy_estimates import AttackResults

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with 'pip install wandb' for experiment tracking.")
from utils.clipbkd import craft_clipbkd, choose_worstcase_label

import torch.nn.functional as F
import torchvision.transforms.v2 as v2

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Enable performance optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def fgsm_attack(model, X, y, epsilon=0.1, max_iter=10, alpha=0.01):
    """
    Perform iterative FGSM attack on the input X to make it misclassified as target class y.
    """
    model.eval()
    X_adv = X.clone().detach().requires_grad_(True)
    best_adv = X_adv.detach().clone()
    best_confidence = -float('inf')
    
    for i in range(max_iter):
        output = model(X_adv)
        _, predicted = torch.max(output, 1)
        if predicted != y:
            return X_adv.detach(), i + 1
            
        confidence = F.softmax(output, dim=1)[0, y].item()
        if confidence > best_confidence:
            best_confidence = confidence
            best_adv = X_adv.detach().clone()
        
        loss = F.cross_entropy(output, y)
        model.zero_grad()
        loss.backward()
        
        data_grad = X_adv.grad.data
        sign_data_grad = data_grad.sign()
        X_adv = X_adv.detach() + alpha * sign_data_grad
        delta = X_adv - X
        delta = torch.clamp(delta, -epsilon, epsilon)
        X_adv = torch.clamp(X + delta, 0, 1).detach().requires_grad_(True)
    
    return best_adv, max_iter


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


def setup_device():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        # Set optimal CUDA settings
        torch.cuda.set_device(device)
    return device


class IndexedTensorDataset(Dataset):
    """A dataset that includes the index of each sample."""
    def __init__(self, *tensors):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        
    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors) + (index,)
        
    def __len__(self):
        return self.tensors[0].size(0)


class AugmentedDataset(Dataset):
    """
    Dataset wrapper that applies augmentation multiplicity.
    Each sample is repeated aug_mult times with different augmentations.
    """
    def __init__(self, X, y, aug_fn, aug_mult=1, indices=None):
        self.X = X
        self.y = y
        self.aug_fn = aug_fn
        self.aug_mult = aug_mult
        self.indices = indices
        
    def __getitem__(self, index):
        # Get the original sample
        x = self.X[index]
        y = self.y[index]
        idx = int(self.indices[index]) if self.indices is not None else index
        
        if self.aug_mult > 1 and self.aug_fn is not None:
            # Apply augmentation aug_mult times and stack
            augmented = torch.stack([self.aug_fn(x) for _ in range(self.aug_mult)])
            return augmented, y, idx
        else:
            # Return with extra dimension for consistency
            return x.unsqueeze(0), y, idx
    
    def __len__(self):
        return len(self.X)


class DefenseDataset(Dataset):
    """
    Dataset wrapper that supports dropping samples for the defense mechanism.
    Samples with drop_mask[i] == True are excluded from training.
    """
    def __init__(self, X, y, drop_mask=None):
        self.X = X
        self.y = y
        self.drop_mask = drop_mask if drop_mask is not None else np.zeros(len(X), dtype=bool)
        self._update_active_indices()
    
    def _update_active_indices(self):
        """Update the list of active (non-dropped) indices."""
        self.active_indices = np.where(~self.drop_mask)[0]
    
    def update_drop_mask(self, drop_mask):
        """Update the drop mask and refresh active indices."""
        self.drop_mask = drop_mask
        self._update_active_indices()
    
    def drop_samples(self, indices_to_drop):
        """Mark specific samples as dropped."""
        for idx in indices_to_drop:
            self.drop_mask[idx] = True
        self._update_active_indices()
    
    def __getitem__(self, index):
        # Map the sequential index to the actual data index
        actual_idx = self.active_indices[index]
        return self.X[actual_idx], self.y[actual_idx], actual_idx
    
    def __len__(self):
        return len(self.active_indices)
    
    def get_original_length(self):
        return len(self.X)


def make_opacus_compatible(model):
    """Make a model compatible with Opacus by replacing incompatible layers."""
    if not ModuleValidator.is_valid(model):
        model = ModuleValidator.fix(model)
    return model


def compute_per_sample_losses(model, X, y, criterion, batch_size=512, device='cuda'):
    """
    Compute per-sample losses for all samples in the dataset.
    Used by the defense to identify high-loss (suspicious) samples.
    """
    model.eval()
    losses = np.zeros(len(X))
    
    loader = DataLoader(
        IndexedTensorDataset(X, y),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
        for curr_X, curr_y, indices in loader:
            curr_X, curr_y = curr_X.to(device, non_blocking=True), curr_y.to(device, non_blocking=True)
            output = model(curr_X)
            
            # Compute per-sample loss (no reduction)
            per_sample_loss = F.cross_entropy(output, curr_y, reduction='none')
            losses[indices.cpu().numpy()] = per_sample_loss.cpu().numpy()
    
    model.train()
    return losses


def train_model_opacus(model_name, X, y, X_target, y_target, epsilon, delta, max_grad_norm, 
                       n_epochs, lr, batch_size, init_model=None, out_dim=10, aug_mult=1,
                       defense=False, defense_k=5, use_wandb=False, wandb_run=None, 
                       world='in', rep=0, max_physical_batch_size=None, optimizer_name='sgd',
                       early_stopping_patience=None):
    """
    Train a model using Opacus for DP-SGD.
    
    Args:
        model_name: Name of the model architecture
        X: Training features
        y: Training labels
        X_target: Target sample features (canary)
        y_target: Target sample labels
        epsilon: Privacy budget (None for non-private training)
        delta: Privacy parameter delta
        max_grad_norm: Maximum gradient norm for clipping
        n_epochs: Number of training epochs
        lr: Learning rate
        batch_size: Batch size for training
        init_model: Initial model weights (optional)
        out_dim: Output dimension (number of classes)
    
    Returns:
        Trained model
    """
    device = setup_device()
    
    # Initialize model
    if init_model is None:
        model = Models[model_name](X.shape, out_dim=out_dim)
        if model_name == 'cnn':
            xavier_init_model(model)
        else:
            init_wideresnet(model)
    else:
        model = copy.deepcopy(init_model)
    
    # Make model Opacus-compatible (replaces BatchNorm with GroupNorm, etc.)
    model = make_opacus_compatible(model)
    model = model.to(device)
    model.train()
    
    # Note: torch.compile is applied in main() before calling this function
    # Opacus wraps the model, so we can't compile here
    
    criterion = nn.CrossEntropyLoss(reduction="none")
    
    use_private = epsilon is not None and max_grad_norm is not None
    
    # Setup optimizer
    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr)
    
    # Setup augmentation function if aug_mult > 1
    aug_fn = None
    if aug_mult > 1 and len(X.shape) > 2:
        aug_fn = AugmentationFunction(X.shape[2], X.shape[1])
        print(f"Using augmentation multiplicity: {aug_mult}")
    
    # Create DataLoader
    if aug_mult > 1 and aug_fn is not None:
        # Use custom dataset with augmentation
        dataset = AugmentedDataset(X, y, aug_fn, aug_mult, indices=np.arange(len(X)))
        
        # Custom collate function to handle augmented batches
        def collate_augmented(batch):
            # batch is list of (augmented_x [aug_mult, C, H, W], y, idx)
            xs = torch.stack([item[0] for item in batch])  # [B, aug_mult, C, H, W]
            ys = torch.tensor([item[1] for item in batch])
            idxs = torch.tensor([item[2] for item in batch])
            return xs, ys, idxs
        
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0,
            collate_fn=collate_augmented,
        )
    else:
        if defense:
            dataset = TensorDataset(X, y, torch.arange(len(X), dtype=torch.long))
        else:
            dataset = TensorDataset(X, y)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0,
            pin_memory=True,
        )
    
    # Defense state
    # 0 = normal, 1 = apply gradient-ascent once, 2 = dropped
    drop_mask = np.zeros(len(X), dtype=np.int8)
    scores = np.zeros(len(X), dtype=np.float32)
    use_private = epsilon is not None and max_grad_norm is not None
    print('use_private', use_private)
    privacy_engine = None
    canary_dropped_epoch = None  # Track when canary is dropped
    canary_idx = len(X) - 1  # Canary is always the last sample
    
    # We'll recreate the dataloader each epoch if defense is enabled
    # to account for dropped samples
    def create_loader_and_engine(X_active, y_active, active_indices=None):
        """Create a new dataloader and optionally wrap with privacy engine."""
        nonlocal privacy_engine
        if active_indices is None:
            active_indices = np.arange(len(X_active))
        
        if aug_mult > 1 and aug_fn is not None:
            # Use augmented dataset
            ds = AugmentedDataset(X_active, y_active, aug_fn, aug_mult, indices=active_indices)
            
            def collate_augmented(batch):
                xs = torch.stack([item[0] for item in batch])
                ys = torch.tensor([item[1] for item in batch])
                idxs = torch.tensor([item[2] for item in batch])
                return xs, ys, idxs
            
            ldr = DataLoader(
                ds,
                batch_size=min(batch_size, len(ds)),
                shuffle=True,
                drop_last=True,
                num_workers=0,
                collate_fn=collate_augmented,
            )
        else:
            if defense:
                ds = TensorDataset(X_active, y_active, torch.tensor(active_indices, dtype=torch.long))
            else:
                ds = TensorDataset(X_active, y_active)
            ldr = DataLoader(
                ds,
                batch_size=min(batch_size, len(ds)),
                shuffle=True,
                drop_last=True,
                num_workers=0,
                pin_memory=True,
            )
        
        return ldr
    
    # Initial loader setup
    loader = create_loader_and_engine(X, y)
    
    # Setup Opacus PrivacyEngine if epsilon is specified
    if use_private:
        # Compute noise multiplier the same way as audit_model.py
        sample_rate = batch_size / len(X)
        noise_multiplier = get_noise_multiplier(
            target_epsilon=epsilon,
            target_delta=delta,
            sample_rate=sample_rate,
            epochs=n_epochs,
            accountant='rdp'
        )
        print(f"Using noise multiplier: {noise_multiplier:.4f} (sample_rate={sample_rate:.4f})")
        print(f"Dataset size: {len(X)}, Batch size: {batch_size}, Expected batches/epoch: {len(X) // batch_size}")
        
        privacy_engine = PrivacyEngine()
        
        # Use make_private with pre-computed noise_multiplier for consistency with audit_model.py
        model, optimizer, loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=loader,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            poisson_sampling=False,  # Use fixed batch size like audit_model.py
        )
    
    # Set max physical batch size for gradient accumulation
    if max_physical_batch_size is None:
        max_physical_batch_size = batch_size
    
    # Early stopping state
    best_loss = float('inf')
    patience_counter = 0

    def _log_dp_clip_stats(batch_idx: int, prefix: str = ""):
        if not use_private:
            return
        if batch_idx % 10 != 0:
            return
        per_sample_norm_sqs = None
        for p in model.parameters():
            if hasattr(p, 'grad_sample') and p.grad_sample is not None:
                gs = p.grad_sample
                gs_flat = gs.view(gs.shape[0], -1)
                norm_sqs = (gs_flat ** 2).sum(dim=1)
                per_sample_norm_sqs = norm_sqs if per_sample_norm_sqs is None else (per_sample_norm_sqs + norm_sqs)

        if per_sample_norm_sqs is None:
            print(f"{prefix}Batch {batch_idx}: grad_sample missing (Opacus not attached or grad_sample cleared)")
            return

        per_sample_norms = per_sample_norm_sqs.sqrt()
        max_before = per_sample_norms.max().item()
        mean_before = per_sample_norms.mean().item()
        num_clipped = int((per_sample_norms > float(max_grad_norm)).sum().item())
        total = int(per_sample_norms.numel())

        clipped_norms = torch.clamp(per_sample_norms, max=float(max_grad_norm))
        max_after = clipped_norms.max().item()
        mean_after = clipped_norms.mean().item()

        print(
            f"{prefix}Batch {batch_idx}: per-sample grad L2 norm before clip max={max_before:.4f} mean={mean_before:.4f} "
            f"| clipped {num_clipped}/{total} | after clip max={max_after:.4f} mean={mean_after:.4f} "
            f"| max_grad_norm={max_grad_norm}"
        )
    
    # Training loop
    for epoch in range(n_epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        n_batches = 0
        
        # Get active samples (not dropped by defense)
        active_mask = (drop_mask != 2)
        n_active = int(active_mask.sum())
        
        if defense:
            print(f"Epoch {epoch} (Active samples: {n_active}/{len(X)})", end='', flush=True)
        
        if aug_mult > 1 and aug_fn is not None:
            # Training with augmentation multiplicity
            # Use BatchMemoryManager for gradient accumulation if needed
            if use_private and max_physical_batch_size < batch_size:
                with BatchMemoryManager(
                    data_loader=loader,
                    max_physical_batch_size=max_physical_batch_size,
                    optimizer=optimizer
                ) as memory_safe_loader:
                    for batch_idx, (curr_X, curr_y, idxs) in enumerate(memory_safe_loader):
                        B, A, C, H, W = curr_X.shape
                        curr_X_flat = curr_X.view(B * A, C, H, W).to(device, non_blocking=True)
                        curr_y_rep = curr_y.repeat_interleave(A).to(device, non_blocking=True)
                        idxs = idxs.to(device, non_blocking=True)
                        
                        optimizer.zero_grad(set_to_none=True)
                        output = model(curr_X_flat)
                        loss_per_view = criterion(output, curr_y_rep)  # [B*A]
                        loss = loss_per_view.view(B, A).mean(dim=1).mean()
                        loss.backward()

                        _log_dp_clip_stats(batch_idx, prefix="[aug|mem] ")
                        
                        # Aggregate per-sample gradients across augmentations
                        for param in model.parameters():
                            if hasattr(param, 'grad_sample') and param.grad_sample is not None:
                                gs = param.grad_sample
                                gs_reshaped = gs.view(B, A, *gs.shape[1:])
                                gs_agg = gs_reshaped.mean(dim=1)
                                param.grad_sample = gs_agg

                        if defense and use_private:
                            gs_list = []
                            for p in model.parameters():
                                if hasattr(p, 'grad_sample') and p.grad_sample is not None:
                                    gs_list.append(p.grad_sample.view(p.grad_sample.shape[0], -1))
                            if len(gs_list) > 0:
                                per_sample_flat_grads = torch.cat(gs_list, dim=1)
                                all_norms = torch.zeros_like(curr_y, dtype=torch.float32)
                                for cls in torch.unique(curr_y):
                                    cls_mask = (curr_y == cls)
                                    if cls_mask.any():
                                        all_norms[cls_mask] = per_sample_flat_grads[cls_mask].norm(float('inf'), dim=1)
                                scores[idxs.detach().cpu().numpy()] = all_norms.detach().cpu().numpy()

                            ascent_mask = (torch.from_numpy(drop_mask[idxs.detach().cpu().numpy()]).to(device) == 1)
                            if ascent_mask.any():
                                for p in model.parameters():
                                    if hasattr(p, 'grad_sample') and p.grad_sample is not None:
                                        p.grad_sample[ascent_mask] *= -1
                        
                        optimizer.step()

                        if defense:
                            ascent_idxs = idxs.detach().cpu().numpy()[
                                (drop_mask[idxs.detach().cpu().numpy()] == 1)
                            ]
                            drop_mask[ascent_idxs] = 2
                        epoch_loss += loss.item()
                        n_batches += 1
            else:
                for batch_idx, (curr_X, curr_y, idxs) in enumerate(loader):
                    B, A, C, H, W = curr_X.shape
                    curr_X_flat = curr_X.view(B * A, C, H, W).to(device, non_blocking=True)
                    curr_y_rep = curr_y.repeat_interleave(A).to(device, non_blocking=True)
                    idxs = idxs.to(device, non_blocking=True)
                    
                    optimizer.zero_grad(set_to_none=True)
                    output = model(curr_X_flat)
                    loss_per_view = criterion(output, curr_y_rep)  # [B*A]
                    loss = loss_per_view.view(B, A).mean(dim=1).mean()
                    loss.backward()

                    _log_dp_clip_stats(batch_idx, prefix="[aug] ")
                    
                    # Aggregate per-sample gradients across augmentations (Opacus only)
                    if use_private:
                        for param in model.parameters():
                            if hasattr(param, 'grad_sample') and param.grad_sample is not None:
                                gs = param.grad_sample
                                gs_reshaped = gs.view(B, A, *gs.shape[1:])
                                gs_agg = gs_reshaped.mean(dim=1)
                                param.grad_sample = gs_agg

                    if defense and use_private:
                        gs_list = []
                        for p in model.parameters():
                            if hasattr(p, 'grad_sample') and p.grad_sample is not None:
                                gs_list.append(p.grad_sample.view(p.grad_sample.shape[0], -1))
                        if len(gs_list) > 0:
                            per_sample_flat_grads = torch.cat(gs_list, dim=1)
                            all_norms = torch.zeros_like(curr_y, dtype=torch.float32)
                            for cls in torch.unique(curr_y):
                                cls_mask = (curr_y == cls)
                                if cls_mask.any():
                                    all_norms[cls_mask] = per_sample_flat_grads[cls_mask].norm(float('inf'), dim=1)
                            scores[idxs.detach().cpu().numpy()] = all_norms.detach().cpu().numpy()

                        ascent_mask = (torch.from_numpy(drop_mask[idxs.detach().cpu().numpy()]).to(device) == 1)
                        if ascent_mask.any():
                            for p in model.parameters():
                                if hasattr(p, 'grad_sample') and p.grad_sample is not None:
                                    p.grad_sample[ascent_mask] *= -1
                    
                    optimizer.step()

                    if defense:
                        ascent_idxs = idxs.detach().cpu().numpy()[
                            (drop_mask[idxs.detach().cpu().numpy()] == 1)
                        ]
                        drop_mask[ascent_idxs] = 2
                    epoch_loss += loss.item()
                    n_batches += 1
        else:
            # Standard training without augmentation multiplicity
            # Use BatchMemoryManager for gradient accumulation if needed
            if use_private and max_physical_batch_size < batch_size:
                with BatchMemoryManager(
                    data_loader=loader,
                    max_physical_batch_size=max_physical_batch_size,
                    optimizer=optimizer
                ) as memory_safe_loader:
                    for batch_idx, batch in enumerate(memory_safe_loader):
                        if defense:
                            curr_X, curr_y, idxs = batch
                            idxs = idxs.to(device, non_blocking=True)
                        else:
                            curr_X, curr_y = batch
                            idxs = None
                        curr_X, curr_y = curr_X.to(device, non_blocking=True), curr_y.to(device, non_blocking=True)
                        
                        optimizer.zero_grad(set_to_none=True)
                        output = model(curr_X)
                        loss = criterion(output, curr_y).mean()
                        loss.backward()

                        _log_dp_clip_stats(batch_idx, prefix="[mem] ")

                        if defense and use_private and idxs is not None:
                            gs_list = []
                            for p in model.parameters():
                                if hasattr(p, 'grad_sample') and p.grad_sample is not None:
                                    gs_list.append(p.grad_sample.view(p.grad_sample.shape[0], -1))
                            if len(gs_list) > 0:
                                per_sample_flat_grads = torch.cat(gs_list, dim=1)
                                all_norms = torch.zeros_like(curr_y, dtype=torch.float32)
                                for cls in torch.unique(curr_y):
                                    cls_mask = (curr_y == cls)
                                    if cls_mask.any():
                                        all_norms[cls_mask] = per_sample_flat_grads[cls_mask].norm(float('inf'), dim=1)
                                scores[idxs.detach().cpu().numpy()] = all_norms.detach().cpu().numpy()

                            ascent_mask = (torch.from_numpy(drop_mask[idxs.detach().cpu().numpy()]).to(device) == 1)
                            if ascent_mask.any():
                                for p in model.parameters():
                                    if hasattr(p, 'grad_sample') and p.grad_sample is not None:
                                        p.grad_sample[ascent_mask] *= -1
                        
                        optimizer.step()

                        if defense and idxs is not None:
                            ascent_idxs = idxs.detach().cpu().numpy()[
                                (drop_mask[idxs.detach().cpu().numpy()] == 1)
                            ]
                            drop_mask[ascent_idxs] = 2
                        
                        epoch_loss += loss.item()
                        n_batches += 1
            else:
                for batch_idx, batch in enumerate(loader):
                    if defense:
                        curr_X, curr_y, idxs = batch
                        idxs = idxs.to(device, non_blocking=True)
                    else:
                        curr_X, curr_y = batch
                        idxs = None
                    curr_X, curr_y = curr_X.to(device, non_blocking=True), curr_y.to(device, non_blocking=True)
                    
                    optimizer.zero_grad(set_to_none=True)
                    output = model(curr_X)
                    loss = criterion(output, curr_y).mean()
                    loss.backward()

                    _log_dp_clip_stats(batch_idx, prefix="")

                    if defense and use_private and idxs is not None:
                        gs_list = []
                        for p in model.parameters():
                            if hasattr(p, 'grad_sample') and p.grad_sample is not None:
                                gs_list.append(p.grad_sample.view(p.grad_sample.shape[0], -1))
                        if len(gs_list) > 0:
                            per_sample_flat_grads = torch.cat(gs_list, dim=1)
                            all_norms = torch.zeros_like(curr_y, dtype=torch.float32)
                            for cls in torch.unique(curr_y):
                                cls_mask = (curr_y == cls)
                                if cls_mask.any():
                                    all_norms[cls_mask] = per_sample_flat_grads[cls_mask].norm(float('inf'), dim=1)
                            scores[idxs.detach().cpu().numpy()] = all_norms.detach().cpu().numpy()

                        ascent_mask = (torch.from_numpy(drop_mask[idxs.detach().cpu().numpy()]).to(device) == 1)
                        if ascent_mask.any():
                            for p in model.parameters():
                                if hasattr(p, 'grad_sample') and p.grad_sample is not None:
                                    p.grad_sample[ascent_mask] *= -1
                    optimizer.step()

                    if defense and idxs is not None:
                        ascent_idxs = idxs.detach().cpu().numpy()[
                            (drop_mask[idxs.detach().cpu().numpy()] == 1)
                        ]
                        drop_mask[ascent_idxs] = 2
                    
                    epoch_loss += loss.item()
                    n_batches += 1
        
        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / n_batches if n_batches > 0 else 0
        
        if use_private:
            eps_spent = privacy_engine.get_epsilon(delta)
            print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Time={epoch_time:.2f}s, ε={eps_spent:.2f}, Batches={n_batches}")
            
            # Log to wandb
            if use_wandb and wandb_run is not None:
                wandb_run.log({
                    f'{world}/rep_{rep}/loss': avg_loss,
                    f'{world}/rep_{rep}/epsilon_spent': eps_spent,
                    f'{world}/rep_{rep}/epoch_time': epoch_time,
                    f'{world}/rep_{rep}/active_samples': n_active,
                    'epoch': epoch,
                })
        else:
            if defense:
                print(f" | Loss={avg_loss:.4f}, Time={epoch_time:.2f}s")
            else:
                print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Time={epoch_time:.2f}s")
            
            # Log to wandb
            if use_wandb and wandb_run is not None:
                wandb_run.log({
                    f'{world}/rep_{rep}/loss': avg_loss,
                    f'{world}/rep_{rep}/epoch_time': epoch_time,
                    f'{world}/rep_{rep}/active_samples': n_active,
                    'epoch': epoch,
                })
        
        # Early stopping check
        if early_stopping_patience is not None:
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping triggered at epoch {epoch} (patience={early_stopping_patience})")
                    break
        
        # Defense: identify and mark samples for one-step gradient ascent (then drop)
        if defense and use_private:
            unique_classes = torch.unique(y).cpu().numpy()
            samples_to_mark = []

            for cls in unique_classes:
                cls_mask = (y.cpu().numpy() == cls) & (drop_mask != 2)
                cls_indices = np.where(cls_mask)[0]

                if len(cls_indices) == 0:
                    continue

                cls_scores = scores[cls_indices]
                k = min(defense_k, len(cls_indices))
                topk_local_indices = np.argsort(cls_scores)[-k:]
                topk_global_indices = cls_indices[topk_local_indices]

                samples_to_mark.extend(topk_global_indices)

                if canary_idx in topk_global_indices and canary_dropped_epoch is None:
                    canary_dropped_epoch = epoch
                    print(f"\n[INFO] Canary (index {canary_idx}) was marked by defense at epoch {epoch}!")

                    if use_wandb and wandb_run is not None:
                        wandb_run.log({
                            f'{world}/rep_{rep}/canary_dropped_epoch': epoch,
                        })

            for idx in samples_to_mark:
                if drop_mask[idx] == 0:
                    drop_mask[idx] = 1

            # Recreate loader with remaining (non-dropped) samples, preserving original indices
            active_indices = np.where(drop_mask != 2)[0]
            if len(active_indices) > 0:
                X_active = X[active_indices]
                y_active = y[active_indices]
                loader = create_loader_and_engine(X_active, y_active, active_indices=active_indices)

                if use_private:
                    loader = privacy_engine._prepare_data_loader(loader, distributed=False, poisson_sampling=False)
    
    # Report canary drop status
    if defense:
        if canary_dropped_epoch is not None:
            print(f"[DEFENSE] Canary was dropped at epoch {canary_dropped_epoch}/{n_epochs}")
        else:
            print(f"[DEFENSE] Canary was NOT dropped during training")
    
    return model, canary_dropped_epoch if defense else None


def test_model(model, X, y, batch_size=512):
    """Evaluate model accuracy on a dataset."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    
    test_loader = DataLoader(
        TensorDataset(X, y), 
        batch_size=batch_size, 
        shuffle=False,
        pin_memory=True,
        num_workers=0,
    )
    
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
        for curr_X, curr_y in test_loader:
            curr_X = curr_X.to(device, non_blocking=True)
            curr_y = curr_y.to(device, non_blocking=True)
            
            output = model(curr_X)
            correct += (output.argmax(1) == curr_y).sum().item()
            total += curr_y.size(0)
    
    model.train()
    return correct / total if total > 0 else 0.0


def save_checkpoint(out_folder, outputs, losses, all_losses, train_set_accs, test_set_accs, fit_world_only):
    """Save checkpoint"""
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
    """Load checkpoint if resume is set to True and previous checkpoint exists"""
    outputs = {'out': [], 'in': []}
    losses = {'out': [], 'in': []}
    all_losses = {'in': [], 'out': []}
    train_set_accs = []
    test_set_accs = []
    
    if os.path.exists(out_folder) and resume:
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
        os.makedirs(out_folder, exist_ok=True)
        save_checkpoint(out_folder, outputs, losses, all_losses, train_set_accs, test_set_accs, fit_world_only)
    
    return outputs, losses, all_losses, train_set_accs, test_set_accs


def main():
    parser = argparse.ArgumentParser(description='Audit DP-SGD using Opacus')
    
    parser.add_argument('--data_name', type=str, default='mnist', 
                        help='dataset to use (mnist, cifar10, cifar100)')
    parser.add_argument('--model_name', type=str, default='cnn', 
                        choices=list(Models.keys()), help='model to audit')
    parser.add_argument('--n_reps', type=int, default=200, 
                        help='number of models to train')
    parser.add_argument('--n_df', type=int, default=0, 
                        help='|D| (0 => use full dataset)')
    parser.add_argument('--n_epochs', type=int, default=100, 
                        help='number of epochs to train for')
    parser.add_argument('--early_stopping', type=int, default=None,
                        help='early stopping patience (number of epochs without improvement)')
    parser.add_argument('--lr', type=float, default=1.33e-4, 
                        help='learning rate')
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam'],
                        help='optimizer to use (sgd or adam)')
    parser.add_argument('--max_grad_norm', type=float, default=1, 
                        help='gradient clipping norm')
    parser.add_argument('--epsilon', type=float, default=10.0, 
                        help='privacy parameter, epsilon')
    parser.add_argument('--delta', type=float, default=1e-5, 
                        help='privacy parameter, delta')
    parser.add_argument('--target_type', type=str, default='blank', 
                        help='sample to use as target (blank, clipbkd, badnets, or path)')
    parser.add_argument('--blank_alpha', type=float, default=0.0, 
                        help='interpolation factor for blank target')
    parser.add_argument('--seed', type=int, default=0, 
                        help='seed for reproducibility')
    parser.add_argument('--out', type=str, default='opacus_results/', 
                        help='folder to write results to')
    parser.add_argument('--fixed_init', type=str, nargs='?', default=None, const='', 
                        help='initialize all models to the same weights')
    parser.add_argument('--batch_size', type=int, default=4000, 
                        help='batch size for training')
    parser.add_argument('--resume', action='store_true', 
                        help='skip experiment if results are present')
    parser.add_argument('--fit_world_only', type=str, default=None, 
                        choices=['in', 'out'], help='just fit models in world and calculate losses')
    parser.add_argument('--alpha', type=float, default=0.05, 
                        help='significance level for empirical eps estimation')
    parser.add_argument('--badnets_label', type=int, default=-1, 
                        help='assign badnets poison this label')
    parser.add_argument('--view_badnets', action='store_true')
    parser.add_argument('--holdout_audit', action='store_true')
    parser.add_argument('--aug_mult', type=int, default=1,
                        help='augmentation multiplier (default: 1)')
    parser.add_argument('--max_physical_batch_size', type=int, default=None,
                        help='max physical batch size for gradient accumulation (default: same as batch_size)')
    parser.add_argument('--linear_threshold', action='store_true',
                        help='use logistic regression to find optimal threshold instead of exhaustive search')
    parser.add_argument('--defense', action='store_true',
                        help='use filtering defense during audit')
    parser.add_argument('--defense_k', type=int, default=5,
                        help='number of top samples to drop per class per epoch (default: 5)')
    parser.add_argument('--wandb', action='store_true',
                        help='enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='opacus-audit',
                        help='Weights & Biases project name')
    parser.add_argument('--wandb_name', type=str, default=None,
                        help='Weights & Biases run name (auto-generated if not provided)')
    parser.add_argument('--compile', action='store_true',
                        help='use torch.compile for model optimization (requires PyTorch 2.0+)')
    
    args = parser.parse_args()
    
    if args.max_grad_norm == -1:
        args.max_grad_norm = None
    
    device = setup_device()
    print(f"Using device: {device}")
    
    # Initialize Weights & Biases
    wandb_run = None
    if args.wandb:
        if not WANDB_AVAILABLE:
            print("Warning: --wandb flag set but wandb is not installed. Skipping wandb logging.")
        else:
            run_name = args.wandb_name or f"{args.data_name}_{args.model_name}_eps{args.epsilon}"
            if args.defense:
                run_name += "_defense"
            
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=run_name,
                config={
                    'data_name': args.data_name,
                    'model_name': args.model_name,
                    'n_reps': args.n_reps,
                    'n_epochs': args.n_epochs,
                    'lr': args.lr,
                    'max_grad_norm': args.max_grad_norm,
                    'epsilon': args.epsilon,
                    'delta': args.delta,
                    'batch_size': args.batch_size,
                    'target_type': args.target_type,
                    'seed': args.seed,
                    'aug_mult': args.aug_mult,
                    'defense': args.defense,
                    'defense_k': args.defense_k,
                    'linear_threshold': args.linear_threshold,
                }
            )
            print(f"Weights & Biases initialized: {wandb_run.url}")
    
    # Reproducibility
    print('Setting random seeds for reproducibility')
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    out_folder = f'{args.out}/{args.data_name}_{args.model_name}_eps{args.epsilon}'
    os.makedirs(out_folder, exist_ok=True)
    os.makedirs(f'{out_folder}/models', exist_ok=True)
    
    # Load data (define D-)
    print('Loading data')
    if args.n_df == 1:
        X_out, y_out, out_dim = load_data(args.data_name, 1)
    else:
        X_out, y_out, out_dim = load_data(args.data_name, args.n_df - 1)
    
    print(f'Dataset shape: {X_out.shape}, Labels shape: {y_out.shape}')
    
    # Initialize model
    print('Initializing model')
    init_model = None
    if args.fixed_init is not None:
        init_model = Models[args.model_name](X_out.shape, out_dim=out_dim)
        
        if args.fixed_init == '':
            # Initialize model (average-case)
            if args.model_name == 'cnn':
                xavier_init_model(init_model)
            else:
                init_wideresnet(init_model)
        else:
            # Load weights from path (worst-case)
            init_model.load_state_dict(torch.load(args.fixed_init))
            X_out, y_out = X_out[len(X_out) // 2:], y_out[len(y_out) // 2:]
    
    # Apply torch.compile if requested (PyTorch 2.0+)
    if args.compile:
        if hasattr(torch, 'compile'):
            print("Enabling torch.compile optimization...")
            # Note: Opacus may have compatibility issues with torch.compile
            # This is experimental and may not work with all configurations
        else:
            print("Warning: torch.compile not available (requires PyTorch 2.0+)")
    
    # Craft target data point (x_T, y_T)
    print('Crafting target data point')
    if args.target_type == 'blank':
        blank_img = torch.zeros_like(X_out[[0]])
        if args.blank_alpha > 0:
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
        target_X = X_out[-1].clone()
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
        target_X, target_y = craft_clipbkd(X_out, init_model)
    elif args.target_type == 'fgsm':
        print("Preparing FGSM attack by training a model on the available data...")
        
        fgsm_model = Models[args.model_name](X_out.shape, out_dim=out_dim).to(device)
        if args.model_name == 'cnn':
            xavier_init_model(fgsm_model)
        else:
            init_wideresnet(fgsm_model)
        
        print("Training FGSM model (non-private)...")
        fgsm_model, _ = train_model_opacus(
            model_name=args.model_name,
            X=X_out,
            y=y_out,
            X_target=None,
            y_target=None,
            epsilon=None,
            delta=None,
            max_grad_norm=None,
            n_epochs=args.n_epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            init_model=fgsm_model,
            out_dim=out_dim,
            aug_mult=1,  # No augmentation for FGSM model training
            defense=False,
        )
        print("FGSM model training completed")
        
        original_X = X_out[-1].unsqueeze(0).to(device)
        original_y = y_out[-1].unsqueeze(0).to(device)
        
        num_classes = out_dim
        target_class = (original_y + 1) % num_classes
        
        print(f"Performing FGSM attack (original: {original_y.item()}, target: {target_class.item()})")
        
        target_X, iters_used = fgsm_attack(
            fgsm_model, original_X, target_class,
            epsilon=0.1, max_iter=20, alpha=0.01
        )
        target_y = target_class
        print(f"FGSM attack completed in {iters_used} iterations")
        
        if not target_X.is_cpu:
            target_X = target_X.cpu()
        if not target_y.is_cpu:
            target_y = target_y.cpu()
        
        del fgsm_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    elif os.path.exists(args.target_type):
        # Check if it's a .pt file (defense-aware canary) or .npy file
        if args.target_type.endswith('.pt'):
            canary_data = torch.load(args.target_type)
            target_X = canary_data['canary'].unsqueeze(0)
            target_y = torch.tensor([canary_data['audit_label']])
            print(f"Loaded defense-aware canary: true_label={canary_data['true_label']}, audit_label={canary_data['audit_label']}")
            
            # Use the same init_model if available in canary file
            if 'init_model' in canary_data and args.fixed_init is not None:
                print("Using init_model from canary file for consistency")
                init_model.load_state_dict(canary_data['init_model'])
        else:
            target_X = torch.from_numpy(np.load(args.target_type))
            if init_model is not None:
                target_y = choose_worstcase_label(init_model, target_X)
            else:
                target_y = torch.from_numpy(np.array([9]))
    else:
        raise Exception(f'Target {args.target_type} not found')
    
    # Define D = D- U {(x_T, y_T)}
    X_in, y_in = torch.vstack((X_out[:-1], target_X)), torch.cat((y_out[:-1], target_y))
    X_test, y_test, _ = load_data(args.data_name, None, split='test')
    
    # Pre-convert labels to long type to avoid repeated conversions
    y_out = y_out.long()
    y_in = y_in.long()
    y_test = y_test.long()
    
    print(f'D- size: {len(X_out)}, D size: {len(X_in)}')
    print(f'Target shape: {target_X.shape}, Target label: {target_y}')
    
    # Train models
    print('Training models')
    worlds = [args.fit_world_only] if args.fit_world_only else ['in', 'out']
    models = {'in': [], 'out': []}
    outputs, losses, all_losses, train_set_accs, test_set_accs = resume_checkpoint(
        out_folder, args.fit_world_only, args.resume
    )
    
    # Track canary drop statistics for defense
    canary_drop_epochs = []  # List of epochs when canary was dropped (-1 if not dropped)
    
    for world in worlds:
        curr_X, curr_y = (X_out, y_out) if world == 'out' else (X_in, y_in)
        reps_completed = len(losses[world])
        
        for rep in range(reps_completed, args.n_reps // 2):
            print(f"\n{'='*50}")
            print(f"World: {world}, Rep {rep + 1}/{args.n_reps // 2}")
            print(f"{'='*50}")
            
            model, canary_dropped_epoch = train_model_opacus(
                model_name=args.model_name,
                X=curr_X,
                y=curr_y,
                X_target=target_X,
                y_target=target_y,
                epsilon=args.epsilon,
                delta=args.delta,
                max_grad_norm=args.max_grad_norm,
                n_epochs=args.n_epochs,
                lr=args.lr,
                batch_size=args.batch_size,
                init_model=init_model,
                out_dim=out_dim,
                aug_mult=args.aug_mult,
                defense=args.defense,
                defense_k=args.defense_k,
                use_wandb=args.wandb and wandb_run is not None,
                wandb_run=wandb_run,
                world=world,
                rep=rep,
                max_physical_batch_size=args.max_physical_batch_size,
                optimizer_name=args.optimizer,
                early_stopping_patience=args.early_stopping,
            )
            
            # Track canary drop statistics
            if args.defense and world == 'in':
                if canary_dropped_epoch is not None:
                    canary_drop_epochs.append(canary_dropped_epoch)
                else:
                    canary_drop_epochs.append(-1)  # -1 means not dropped
            
            # Log rep completion to wandb
            if args.wandb and wandb_run is not None:
                wandb_run.log({
                    f'{world}/rep_{rep}/completed': True,
                })
            
            model.eval()
            with torch.no_grad():
                device = next(model.parameters()).device
                target_X_device = target_X.to(device)
                target_y_device = target_y.to(device)
                
                output = model(target_X_device)
                outputs[world].append(output[0].cpu().numpy())
                losses[world].append(-nn.CrossEntropyLoss()(output, target_y_device).cpu().item())
            
            # Save checkpoint after each rep
            save_checkpoint(out_folder, outputs, losses, all_losses, train_set_accs, test_set_accs, args.fit_world_only)
            
            # Get test set accuracy from first 5 reps
            if rep < 5 and world == 'in':
                if len(X_out) > 0:
                    train_set_accs.append(test_model(model, X_in, y_in))
                    print(f'Train set acc: {train_set_accs[-1]:.4f}')
                test_set_accs.append(test_model(model, X_test, y_test))
                print(f'Test set acc: {test_set_accs[-1]:.4f}')
            
            save_checkpoint(out_folder, outputs, losses, all_losses, train_set_accs, test_set_accs, args.fit_world_only)
        
        outputs[world] = np.array(outputs[world])
    
    # Compute empirical epsilon
    if not args.fit_world_only:
        def compute_eps_with_linear_threshold(scores, labels, alpha, delta):
            """
            Use logistic regression to find the optimal threshold for MIA.
            The decision boundary of the logistic regression serves as the threshold.
            """
            scores = np.array(scores).reshape(-1, 1)
            labels = np.array(labels)
            
            # Fit logistic regression
            clf = LogisticRegression(solver='lbfgs', max_iter=1000)
            clf.fit(scores, labels)
            
            # The threshold is where P(y=1|x) = 0.5, i.e., w*x + b = 0 => x = -b/w
            w = clf.coef_[0][0]
            b = clf.intercept_[0]
            if abs(w) > 1e-10:
                threshold = -b / w
            else:
                # If w is near zero, use median as fallback
                threshold = np.median(scores)
            
            # Compute predictions using this threshold
            predictions = (scores.flatten() >= threshold).astype(int)
            
            # Calculate TP, FP, TN, FN
            tp = np.sum((predictions == 1) & (labels == 1))
            fp = np.sum((predictions == 1) & (labels == 0))
            tn = np.sum((predictions == 0) & (labels == 0))
            fn = np.sum((predictions == 0) & (labels == 1))
            
            results = AttackResults(FN=fn, FP=fp, TN=tn, TP=tp)
            emp_eps = compute_eps_lower_single(results, alpha, delta, method='GDP')
            
            return threshold, emp_eps
        
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
            
            mia_scores = np.concatenate([t_losses['in'], t_losses['out']])
            mia_labels = np.concatenate([np.ones_like(t_losses['in']), np.zeros_like(t_losses['out'])])
            
            if args.linear_threshold:
                # Use logistic regression to find threshold
                max_t, emp_eps_loss = compute_eps_with_linear_threshold(
                    mia_scores, mia_labels, args.alpha, args.delta
                )
                print(f"Linear model threshold: {max_t:.6f}")
            else:
                # Exhaustive search over all thresholds
                max_t, emp_eps_loss = compute_eps_lower_from_mia(
                    mia_scores, mia_labels, args.alpha, args.delta, 'GDP', n_procs=1
                )
            
            if args.holdout_audit:
                emp_eps_loss = compute_eps_lower_from_mia_given_t(
                    np.concatenate([holdout_losses['in'], holdout_losses['out']]),
                    np.concatenate([np.ones_like(holdout_losses['in']), np.zeros_like(holdout_losses['out'])]),
                    args.alpha, args.delta, max_t, 'GDP'
                )
            
            return emp_eps_loss, mia_scores, mia_labels
        
        emp_eps_loss, mia_scores, mia_labels = audit_canary(losses, args)
        
        np.save(f'{out_folder}/emp_eps_loss.npy', [emp_eps_loss])
        np.save(f'{out_folder}/mia_scores.npy', mia_scores)
        np.save(f'{out_folder}/mia_labels.npy', mia_labels)
        
        print(f'\n{"="*50}')
        print(f'AUDIT RESULTS')
        print(f'{"="*50}')
        print(f'Theoretical epsilon: {args.epsilon}')
        print(f'Empirical epsilon: {emp_eps_loss}')
        
        # Log final results to wandb
        if args.wandb and wandb_run is not None:
            wandb_run.log({
                'final/theoretical_epsilon': args.epsilon if args.epsilon else 0,
                'final/empirical_epsilon': emp_eps_loss,
                'final/epsilon_gap': (args.epsilon - emp_eps_loss) if args.epsilon else 0,
            })
    
    print(f'Train set accuracy: {np.mean(train_set_accs) * 100:.3f}%')
    print(f'Test set accuracy: {np.mean(test_set_accs) * 100:.3f}%')
    
    # Print defense statistics
    if args.defense and len(canary_drop_epochs) > 0:
        print(f'\n{"="*50}')
        print('DEFENSE STATISTICS')
        print(f'{"="*50}')
        dropped_count = sum(1 for e in canary_drop_epochs if e >= 0)
        total_reps = len(canary_drop_epochs)
        print(f'Canary dropped in {dropped_count}/{total_reps} runs ({100*dropped_count/total_reps:.1f}%)')
        if dropped_count > 0:
            drop_epochs_only = [e for e in canary_drop_epochs if e >= 0]
            print(f'Average drop epoch: {np.mean(drop_epochs_only):.1f}')
            print(f'Earliest drop: epoch {min(drop_epochs_only)}')
            print(f'Latest drop: epoch {max(drop_epochs_only)}')
        
        # Save canary drop statistics
        np.save(f'{out_folder}/canary_drop_epochs.npy', canary_drop_epochs)
        
        # Log defense stats to wandb
        if args.wandb and wandb_run is not None:
            wandb_run.log({
                'defense/canary_drop_rate': dropped_count / total_reps,
                'defense/avg_drop_epoch': np.mean(drop_epochs_only) if dropped_count > 0 else -1,
                'defense/earliest_drop': min(drop_epochs_only) if dropped_count > 0 else -1,
                'defense/latest_drop': max(drop_epochs_only) if dropped_count > 0 else -1,
            })
    
    # Log accuracy to wandb
    if args.wandb and wandb_run is not None:
        wandb_run.log({
            'final/train_accuracy': np.mean(train_set_accs),
            'final/test_accuracy': np.mean(test_set_accs),
        })
        
        # Finish wandb run
        wandb_run.finish()
        print("Weights & Biases run finished.")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nInterrupted by user')
    except Exception as e:
        print(f'\nError: {str(e)}')
        import traceback
        traceback.print_exc()
