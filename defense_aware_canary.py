"""
Defense-Aware Canary Generation using Iterative Perturbation

The goal is to craft a canary that evades the top-k per-class filtering defense.
The defense filters samples by per-sample gradient norm (not loss).

The attacker knows:
- k: number of samples dropped per class per epoch
- n_epochs: total training epochs  
- The model architecture and training procedure (DP-SGD with Opacus)

Strategy:
For each epoch i, we want the canary's per-sample gradient norm to NOT be
in the top-k values among samples of its class. We iteratively perturb the canary
to reduce its gradient norm at each epoch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from copy import deepcopy
from torch.utils.data import DataLoader, TensorDataset

from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from opacus.accountants.utils import get_noise_multiplier
from opacus.grad_sample import GradSampleModule

from models import Models
from utils.data import load_data


def make_opacus_compatible(model):
    """Make model compatible with Opacus (replace BatchNorm, etc.)"""
    if not ModuleValidator.is_valid(model):
        model = ModuleValidator.fix(model)
    return model


def compute_per_sample_gradient_norms(model_state_dict, model_name, in_shape, out_dim, 
                                       X, y, device, batch_size=256):
    """
    Compute the per-sample gradient norm for each sample using Opacus.
    This is exactly what the defense uses to filter samples.
    
    Args:
        model_state_dict: State dict of the model weights
        model_name: Name of the model architecture
        in_shape: Input shape for model creation
        out_dim: Output dimension for model creation
        X: Input data tensor
        y: Labels tensor
        device: Device to use
        batch_size: Batch size for computation
    
    Returns:
        numpy array of per-sample gradient norms
    """
    # Create a completely fresh model and load weights
    fresh_model = Models[model_name](in_shape, out_dim=out_dim)
    fresh_model = make_opacus_compatible(fresh_model)
    # Load state dict with strict=False in case of minor mismatches from Opacus fixes
    fresh_model.load_state_dict(model_state_dict, strict=False)
    fresh_model.to(device)
    fresh_model.train()  # Must be in train mode for GradSampleModule hooks
    
    # Wrap with GradSampleModule
    grad_sample_model = GradSampleModule(fresh_model)
    
    grad_norms = np.zeros(len(X))
    criterion = nn.CrossEntropyLoss(reduction='none')  # Per-sample loss
    
    dataset = TensorDataset(X, y, torch.arange(len(X)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    for batch_X, batch_y, indices in loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        
        grad_sample_model.zero_grad()
        output = grad_sample_model(batch_X)
        loss = criterion(output, batch_y).mean()  # Mean for backward
        loss.backward()
        
        # Compute per-sample gradient norm across all parameters
        batch_grad_norms = []
        for i in range(len(batch_X)):
            sample_grad_norm_sq = 0.0
            for param in grad_sample_model.parameters():
                if hasattr(param, 'grad_sample') and param.grad_sample is not None:
                    # grad_sample has shape [batch_size, *param_shape]
                    sample_grad = param.grad_sample[i]
                    sample_grad_norm_sq += sample_grad.norm() ** 2
            batch_grad_norms.append(np.sqrt(sample_grad_norm_sq.item()))
        
        grad_norms[indices.numpy()] = batch_grad_norms
    
    # Clean up
    del grad_sample_model
    del fresh_model
    
    return grad_norms


def get_top_k_threshold_per_class(values, labels, k):
    """
    For each class, find the threshold value such that k samples
    have values >= threshold (i.e., would be dropped).
    
    Returns:
        dict mapping class -> threshold value
    """
    thresholds = {}
    unique_classes = np.unique(labels)
    
    for cls in unique_classes:
        cls_mask = labels == cls
        cls_values = values[cls_mask]
        
        if len(cls_values) <= k:
            thresholds[cls] = -np.inf  # All would be dropped
        else:
            sorted_values = np.sort(cls_values)[::-1]  # descending
            thresholds[cls] = sorted_values[k - 1] if k > 0 else np.inf
    
    return thresholds


def get_canary_rank_in_class(grad_norms, labels, canary_idx, canary_label):
    """
    Get the rank of the canary's gradient norm within its class.
    Rank 1 = highest gradient norm (most likely to be dropped).
    """
    cls_mask = labels == canary_label
    cls_indices = np.where(cls_mask)[0]
    cls_grad_norms = grad_norms[cls_mask]
    
    canary_grad_norm = grad_norms[canary_idx]
    rank = (cls_grad_norms >= canary_grad_norm).sum()
    
    return rank, len(cls_indices)


def simulate_dpsgd_epoch(model, optimizer, loader, criterion, device, privacy_engine=None):
    """
    Simulate one epoch of DP-SGD training.
    """
    model.train()
    total_loss = 0
    n_batches = 0
    
    for batch_X, batch_y in loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        
        optimizer.zero_grad()
        output = model(batch_X)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches if n_batches > 0 else 0


def create_dpsgd_training_setup(model, X, y, batch_size, lr, epsilon, delta, n_epochs, 
                                 max_grad_norm, device):
    """
    Create a full DP-SGD training setup with Opacus.
    """
    model = make_opacus_compatible(model)
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    # Compute noise multiplier
    sample_rate = batch_size / len(X)
    noise_multiplier = get_noise_multiplier(
        target_epsilon=epsilon,
        target_delta=delta,
        sample_rate=sample_rate,
        epochs=n_epochs,
        accountant='rdp'
    )
    
    privacy_engine = PrivacyEngine()
    model, optimizer, loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
        poisson_sampling=False,
    )
    
    return model, optimizer, loader, privacy_engine


def perturb_canary_to_reduce_grad_norm(
    canary,
    canary_label,
    model_state_dict,
    model_name,
    in_shape,
    out_dim,
    device,
    target_grad_norm,
    n_iterations=20,
    step_size=0.01,
    verbose=False
):
    """
    Perturb the canary to reduce its per-sample gradient norm.
    
    The idea: We want to find a perturbation δ such that the gradient norm
    of the canary (x + δ) is below the threshold.
    
    We use gradient descent on the gradient norm itself:
    - Compute grad_norm = ||∇_θ L(f_θ(x), y)||
    - Compute ∂(grad_norm)/∂x 
    - Update x = x - step_size * sign(∂(grad_norm)/∂x)
    """
    canary = canary.clone().detach().to(device)
    label_tensor = torch.tensor([canary_label]).to(device)
    criterion = nn.CrossEntropyLoss()
    
    for iteration in range(n_iterations):
        # Create fresh model each iteration to avoid hook issues
        fresh_model = Models[model_name](in_shape, out_dim=out_dim)
        fresh_model = make_opacus_compatible(fresh_model)
        fresh_model.load_state_dict(model_state_dict, strict=False)
        fresh_model.to(device)
        fresh_model.train()  # Must be in train mode for GradSampleModule hooks
        
        grad_model = GradSampleModule(fresh_model)
        
        canary.requires_grad = True
        
        # Forward pass
        grad_model.zero_grad()
        output = grad_model(canary.unsqueeze(0))
        loss = criterion(output, label_tensor)
        
        # Compute per-sample gradient norm
        loss.backward(create_graph=True)  # Need graph for second derivative
        
        grad_norm_sq = torch.tensor(0.0, device=device, requires_grad=True)
        for param in grad_model.parameters():
            if hasattr(param, 'grad_sample') and param.grad_sample is not None:
                grad_norm_sq = grad_norm_sq + (param.grad_sample[0] ** 2).sum()
        
        current_grad_norm = torch.sqrt(grad_norm_sq)
        
        if verbose and iteration % 5 == 0:
            print(f"    Iter {iteration}: grad_norm = {current_grad_norm.item():.4f}, target = {target_grad_norm:.4f}")
        
        # Check if we've reached target
        if current_grad_norm.item() <= target_grad_norm:
            if verbose:
                print(f"    Reached target at iteration {iteration}")
            del grad_model, fresh_model
            break
        
        # Compute gradient of grad_norm w.r.t. input
        grad_of_grad_norm = torch.autograd.grad(current_grad_norm, canary, retain_graph=True)[0]
        
        # Update canary to reduce gradient norm
        with torch.no_grad():
            # Use sign of gradient (like FGSM) for more stable updates
            canary = canary - step_size * grad_of_grad_norm.sign()
            canary = canary.detach()
        
        # Clean up
        del grad_model, fresh_model
    
    return canary.detach().cpu()


def perturb_canary_untargeted(
    canary,
    true_label,
    model_state_dict,
    model_name,
    in_shape,
    out_dim,
    device,
    n_iterations=20,
    step_size=0.01,
    verbose=False
):
    """
    Untargeted attack: perturb canary to cause misclassification using gradient ascent.
    Maximizes loss w.r.t. the true label (FGSM-style).
    
    Args:
        canary: Current canary tensor
        true_label: The true/original label to move away from
        model_state_dict: State dict of model weights
        model_name: Model architecture name
        in_shape: Input shape for model
        out_dim: Output dimension
        device: Device
        n_iterations: Number of perturbation iterations
        step_size: Step size for perturbation
        verbose: Print progress
    
    Returns:
        Perturbed canary, predicted (misclassified) label
    """
    canary = canary.clone().detach().to(device)
    label_tensor = torch.tensor([true_label]).to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Create fresh model without Opacus hooks
    model = Models[model_name](in_shape, out_dim=out_dim)
    model = make_opacus_compatible(model)
    model.load_state_dict(model_state_dict, strict=False)
    model.to(device)
    model.eval()
    
    for iteration in range(n_iterations):
        canary.requires_grad = True
        
        output = model(canary.unsqueeze(0))
        loss = criterion(output, label_tensor)
        
        # Check if already misclassified
        pred = output.argmax(dim=1).item()
        
        if verbose and iteration % 5 == 0:
            print(f"    Iter {iteration}: loss = {loss.item():.4f}, pred = {pred}, true = {true_label}")
        
        if pred != true_label:
            if verbose:
                print(f"    Misclassified at iteration {iteration}: pred={pred}, true={true_label}")
            break
        
        # Gradient ascent to maximize loss (cause misclassification)
        model.zero_grad()
        loss.backward()
        
        with torch.no_grad():
            # FGSM: add sign of gradient to maximize loss
            canary = canary + step_size * canary.grad.sign()
            canary = canary.detach()
    
    # Get final prediction
    with torch.no_grad():
        output = model(canary.unsqueeze(0))
        final_pred = output.argmax(dim=1).item()
    
    del model
    return canary.detach().cpu(), final_pred


def train_model_for_t_epochs(model, X, y, t, batch_size, lr, epsilon, delta, 
                             n_total_epochs, max_grad_norm, device):
    """
    Train a fresh model for t epochs using DP-SGD.
    
    Args:
        model: Fresh model to train
        X: Training data (including canary)
        y: Training labels (including canary)
        t: Number of epochs to train
        batch_size: Batch size
        lr: Learning rate
        epsilon: Privacy budget (for full n_total_epochs)
        delta: Privacy parameter
        n_total_epochs: Total epochs (for noise calibration)
        max_grad_norm: Gradient clipping norm
        device: Device
    
    Returns:
        Trained model (unwrapped)
    """
    if t == 0:
        return model
    
    model = make_opacus_compatible(model)
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    # Compute noise multiplier for full training
    sample_rate = batch_size / len(X)
    noise_multiplier = get_noise_multiplier(
        target_epsilon=epsilon,
        target_delta=delta,
        sample_rate=sample_rate,
        epochs=n_total_epochs,
        accountant='rdp'
    )
    
    privacy_engine = PrivacyEngine()
    model, optimizer, loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
        poisson_sampling=False,
    )
    
    # Train for t epochs
    model.train()
    for epoch in range(t):
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
    
    # Return unwrapped model
    return model._module if hasattr(model, '_module') else model


def find_first_drop_epoch(
    canary,
    canary_label,
    X_train,
    y_train,
    model_name,
    init_model,
    n_epochs,
    defense_k,
    batch_size,
    lr,
    epsilon,
    delta,
    max_grad_norm,
    device,
    verbose=True
):
    """
    Find the first epoch where the canary would be dropped by the defense.
    Trains model once for n_epochs and checks at each epoch checkpoint.
    
    Returns:
        first_drop_epoch: The first epoch where canary is in top-k (or n_epochs if never dropped)
        model_state_at_drop: State dict of model at the epoch before first drop (for warm start)
    """
    in_shape = X_train.shape
    out_dim = len(torch.unique(y_train))
    canary_idx = len(X_train)
    
    canary_label_tensor = torch.tensor([canary_label])
    X_with_canary = torch.cat([X_train, canary.unsqueeze(0)], dim=0)
    y_with_canary = torch.cat([y_train, canary_label_tensor], dim=0)
    
    if verbose:
        print("Scanning for first drop epoch (single training run)...")
    
    # Create model and set up for training
    model = Models[model_name](in_shape, out_dim=out_dim)
    model = make_opacus_compatible(model)
    if init_model is not None:
        model.load_state_dict(deepcopy(init_model))
    model.to(device)
    
    # Store states at each epoch (unwrapped, for warm start later)
    model_state_before_drop = deepcopy(model.state_dict())
    
    # Wrap with GradSampleModule for efficient per-sample gradient computation
    grad_sample_model = GradSampleModule(model)
    
    def compute_grad_norms_fast(gsm, X, y):
        """Compute per-sample gradient norms using GradSampleModule directly."""
        gsm.train()
        norms = np.zeros(len(X))
        criterion = nn.CrossEntropyLoss(reduction='mean')
        
        ds = TensorDataset(X, y, torch.arange(len(X)))
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
        
        for bx, by, idxs in dl:
            bx, by = bx.to(device), by.to(device)
            gsm.zero_grad()
            out = gsm(bx)
            loss = criterion(out, by)
            loss.backward()
            
            batch_norms = []
            for i in range(len(bx)):
                norm_sq = 0.0
                for p in gsm.parameters():
                    if hasattr(p, 'grad_sample') and p.grad_sample is not None:
                        gs = p.grad_sample
                        if isinstance(gs, list):
                            gs = gs[0] if len(gs) > 0 else None
                        if gs is not None:
                            norm_sq += gs[i].norm() ** 2
                batch_norms.append(np.sqrt(norm_sq.item() if hasattr(norm_sq, 'item') else norm_sq))
            norms[idxs.numpy()] = batch_norms
        return norms
    
    # Check epoch 0 (before any training)
    grad_norms = compute_grad_norms_fast(grad_sample_model, X_with_canary, y_with_canary)
    canary_grad_norm = grad_norms[canary_idx]
    thresholds = get_top_k_threshold_per_class(grad_norms, y_with_canary.numpy(), defense_k)
    threshold = thresholds[canary_label]
    rank, class_size = get_canary_rank_in_class(grad_norms, y_with_canary.numpy(), canary_idx, canary_label)
    would_be_dropped = canary_grad_norm >= threshold
    
    if verbose:
        print(f"  Epoch 0: grad_norm={canary_grad_norm:.4f}, threshold={threshold:.4f}, "
              f"rank={rank}/{class_size}, dropped={would_be_dropped}")
    
    if would_be_dropped:
        del grad_sample_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        return 0, model_state_before_drop
    
    # Set up DP-SGD training - use the underlying model from GradSampleModule
    # GradSampleModule wraps the model, so we can use it directly for training
    dataset = TensorDataset(X_with_canary, y_with_canary)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.SGD(grad_sample_model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # Train epoch by epoch and check after each
    # Note: We're using GradSampleModule directly which already computes per-sample gradients
    for t in range(1, n_epochs):
        # Save state before training this epoch
        unwrapped = grad_sample_model._module if hasattr(grad_sample_model, '_module') else grad_sample_model
        model_state_before_drop = deepcopy(unwrapped.state_dict())
        
        # Train one epoch with manual gradient clipping and noise (simulating DP-SGD)
        grad_sample_model.train()
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = grad_sample_model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            
            # Clip per-sample gradients and average (simulating DP-SGD clipping)
            # First compute per-sample gradient norms across all parameters
            with torch.no_grad():
                per_sample_norms = torch.zeros(len(batch_X), device=device)
                for p in grad_sample_model.parameters():
                    if hasattr(p, 'grad_sample') and p.grad_sample is not None:
                        gs = p.grad_sample
                        if isinstance(gs, list):
                            gs = gs[0] if len(gs) > 0 else None
                        if gs is not None:
                            per_sample_norms += gs.view(len(batch_X), -1).pow(2).sum(dim=1)
                per_sample_norms = per_sample_norms.sqrt()
                
                # Clip factor per sample
                clip_factor = torch.clamp(max_grad_norm / (per_sample_norms + 1e-6), max=1.0)
                
                # Apply clipping to each parameter and average
                for p in grad_sample_model.parameters():
                    if hasattr(p, 'grad_sample') and p.grad_sample is not None:
                        gs = p.grad_sample
                        if isinstance(gs, list):
                            gs = gs[0] if len(gs) > 0 else None
                        if gs is not None:
                            clipped = gs * clip_factor.view(-1, *([1] * (gs.dim() - 1)))
                            p.grad = clipped.mean(dim=0)
            
            optimizer.step()
        
        # Check if canary would be dropped after this epoch
        grad_norms = compute_grad_norms_fast(grad_sample_model, X_with_canary, y_with_canary)
        
        canary_grad_norm = grad_norms[canary_idx]
        thresholds = get_top_k_threshold_per_class(grad_norms, y_with_canary.numpy(), defense_k)
        threshold = thresholds[canary_label]
        rank, class_size = get_canary_rank_in_class(grad_norms, y_with_canary.numpy(), canary_idx, canary_label)
        would_be_dropped = canary_grad_norm >= threshold
        
        if verbose:
            print(f"  Epoch {t}: grad_norm={canary_grad_norm:.4f}, threshold={threshold:.4f}, "
                  f"rank={rank}/{class_size}, dropped={would_be_dropped}")
        
        if would_be_dropped:
            del grad_sample_model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            return t, model_state_before_drop
    
    # Get final state
    unwrapped = grad_sample_model._module if hasattr(grad_sample_model, '_module') else grad_sample_model
    model_state_before_drop = deepcopy(unwrapped.state_dict())
    
    # Clean up
    del grad_sample_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    if verbose:
        print(f"  Canary never dropped in {n_epochs} epochs!")
    
    return n_epochs, model_state_before_drop


def craft_defense_aware_canary(
    base_canary,
    true_label,
    X_train,
    y_train,
    model_name,
    init_model,
    n_epochs,
    defense_k,
    batch_size=256,
    lr=1e-3,
    epsilon=10.0,
    delta=1e-5,
    max_grad_norm=1.0,
    perturbation_step_size=0.01,
    perturbation_iterations=20,
    warmup_epochs=2,  # How many epochs before first drop to start from
    device='cuda',
    verbose=True
):
    """
    Craft a canary that evades the top-k gradient norm defense for all epochs.
    Uses untargeted attack (gradient ascent) to cause misclassification.
    
    Smart strategy:
      1. First, find the epoch where the original canary first gets dropped
      2. Start perturbations from a few epochs before that (warm start)
      3. For each epoch from start_epoch to n_epochs-1:
         a. Use warm-start model state (pretrained to start_epoch - warmup)
         b. Train additional epochs to reach current epoch
         c. Perturb canary to evade defense at this epoch
    
    Args:
        base_canary: Initial canary tensor
        true_label: True label of the base canary (we will misclassify away from this)
        X_train: Training data (without canary)
        y_train: Training labels (without canary)
        model_name: Model architecture name
        init_model: Initial model state dict (for consistent initialization)
        n_epochs: Number of training epochs to evade
        defense_k: Number of samples dropped per class per epoch
        batch_size: Training batch size
        lr: Learning rate
        epsilon: Privacy budget
        delta: Privacy parameter
        max_grad_norm: Gradient clipping norm
        perturbation_step_size: Step size for canary perturbation
        perturbation_iterations: Max iterations for perturbation per epoch
        warmup_epochs: Number of epochs before first drop to start perturbations
        device: Device to use
        verbose: Print progress
    
    Returns:
        Tuple of (perturbed canary, final misclassified label to use for auditing)
    """
    canary = base_canary.clone().detach()
    
    in_shape = X_train.shape
    out_dim = len(torch.unique(y_train))
    canary_idx = len(X_train)  # Canary will be appended at the end
    
    # Track the current canary label (will be updated after misclassification)
    canary_label = true_label
    
    if verbose:
        print(f"Crafting defense-aware canary (untargeted attack)")
        print(f"  Model: {model_name}")
        print(f"  Epochs to evade: {n_epochs}")
        print(f"  Defense k: {defense_k}")
        print(f"  True label: {true_label}")
        print(f"  Training set size: {len(X_train)}")
        print(f"  Privacy: ε={epsilon}, δ={delta}")
    
    # Step 0: Find the first epoch where canary gets dropped
    first_drop_epoch, warm_start_state = find_first_drop_epoch(
        canary, canary_label, X_train, y_train, model_name, init_model,
        n_epochs, defense_k, batch_size, lr, epsilon, delta, max_grad_norm,
        device, verbose
    )
    
    if first_drop_epoch >= n_epochs:
        if verbose:
            print(f"\nCanary never dropped! No perturbation needed.")
        return canary, canary_label
    
    # Calculate start epoch for perturbations (a few epochs before first drop)
    start_epoch = max(0, first_drop_epoch - warmup_epochs)
    
    if verbose:
        print(f"\n=== Attack Strategy ===")
        print(f"  First drop epoch: {first_drop_epoch}")
        print(f"  Starting perturbations from epoch: {start_epoch}")
        print(f"  Warm-start from epoch: {max(0, first_drop_epoch - 1)}")
    
    # Iterate through epochs starting from start_epoch
    for t in range(start_epoch, n_epochs):
        if verbose:
            print(f"\n=== Epoch {t} ===")
        
        # Create dataset with current canary and its current label
        canary_label_tensor = torch.tensor([canary_label])
        X_with_canary = torch.cat([X_train, canary.unsqueeze(0)], dim=0)
        y_with_canary = torch.cat([y_train, canary_label_tensor], dim=0)
        
        # Use warm-start: load state from epoch before first drop, then train remaining epochs
        model = Models[model_name](in_shape, out_dim=out_dim)
        
        # Calculate how many epochs to train from warm start
        warm_start_epoch = max(0, first_drop_epoch - 1)
        epochs_to_train = t - warm_start_epoch
        
        if epochs_to_train <= 0:
            # Before warm start epoch, train from init
            if init_model is not None:
                model.load_state_dict(deepcopy(init_model))
            if t > 0:
                if verbose:
                    print(f"  Training from init for {t} epochs...")
                model = train_model_for_t_epochs(
                    model, X_with_canary, y_with_canary, t,
                    batch_size, lr, epsilon, delta, n_epochs, max_grad_norm, device
                )
        else:
            # Load warm start state and train remaining epochs
            model.load_state_dict(deepcopy(warm_start_state))
            if verbose:
                print(f"  Warm-starting from epoch {warm_start_epoch}, training {epochs_to_train} more epochs...")
            model = train_model_for_t_epochs(
                model, X_with_canary, y_with_canary, epochs_to_train,
                batch_size, lr, epsilon, delta, n_epochs, max_grad_norm, device
            )
        
        model.to(device)
        
        # Step 1: Untargeted attack - perturb canary to cause misclassification
        if verbose:
            print(f"  Performing untargeted attack (gradient ascent)...")
        
        # Get model state dict for fresh model creation
        model_state = model.state_dict()
        
        canary, new_pred_label = perturb_canary_untargeted(
            canary,
            true_label,  # Always attack away from the TRUE label
            model_state,
            model_name,
            in_shape,
            out_dim,
            device,
            n_iterations=perturbation_iterations,
            step_size=perturbation_step_size,
            verbose=verbose
        )
        
        # Update canary label to the misclassified prediction
        if new_pred_label != canary_label:
            if verbose:
                print(f"  Canary label updated: {canary_label} -> {new_pred_label}")
            canary_label = new_pred_label
        
        # Step 2: Check if canary would be dropped by defense
        # Recompute with updated canary
        canary_label_tensor = torch.tensor([canary_label])
        X_with_canary = torch.cat([X_train, canary.unsqueeze(0)], dim=0)
        y_with_canary = torch.cat([y_train, canary_label_tensor], dim=0)
        
        # model_state already obtained above
        grad_norms = compute_per_sample_gradient_norms(
            model_state, model_name, in_shape, out_dim,
            X_with_canary, y_with_canary, device, batch_size
        )
        
        canary_grad_norm = grad_norms[canary_idx]
        
        # Get threshold for canary's (misclassified) class
        thresholds = get_top_k_threshold_per_class(
            grad_norms, y_with_canary.numpy(), defense_k
        )
        threshold = thresholds[canary_label]
        
        rank, class_size = get_canary_rank_in_class(
            grad_norms, y_with_canary.numpy(), canary_idx, canary_label
        )
        
        would_be_dropped = canary_grad_norm >= threshold
        
        if verbose:
            print(f"  Canary label: {canary_label} (true: {true_label})")
            print(f"  Canary grad norm: {canary_grad_norm:.4f}")
            print(f"  Class {canary_label} threshold (top-{defense_k}): {threshold:.4f}")
            print(f"  Canary rank in class: {rank}/{class_size}")
            print(f"  Would be dropped: {would_be_dropped}")
        
        # Step 3: If canary would be dropped, perturb to reduce gradient norm
        if would_be_dropped:
            if verbose:
                print(f"  Perturbing canary to reduce gradient norm...")
            
            target_norm = threshold * 0.8  # Aim for 80% of threshold
            
            canary = perturb_canary_to_reduce_grad_norm(
                canary,
                canary_label,
                model_state,
                model_name,
                in_shape,
                out_dim,
                device,
                target_grad_norm=target_norm,
                n_iterations=perturbation_iterations,
                step_size=perturbation_step_size,
                verbose=verbose
            )
            
            # Verify new gradient norm
            X_with_canary = torch.cat([X_train, canary.unsqueeze(0)], dim=0)
            new_grad_norms = compute_per_sample_gradient_norms(
                model_state, model_name, in_shape, out_dim,
                X_with_canary, y_with_canary, device, batch_size
            )
            new_canary_grad_norm = new_grad_norms[canary_idx]
            new_rank, _ = get_canary_rank_in_class(
                new_grad_norms, y_with_canary.numpy(), canary_idx, canary_label
            )
            
            if verbose:
                print(f"  New canary grad norm: {new_canary_grad_norm:.4f}")
                print(f"  New rank in class: {new_rank}/{class_size}")
                print(f"  Now evades defense: {new_canary_grad_norm < threshold}")
        
        # Clean up model to free memory
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    if verbose:
        print(f"\n=== Final Result ===")
        print(f"  Final canary label for auditing: {canary_label}")
        print(f"  True label: {true_label}")
    
    return canary, canary_label


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Craft defense-aware canary (untargeted attack)')
    parser.add_argument('--data_name', type=str, default='mnist')
    parser.add_argument('--model_name', type=str, default='cnn')
    parser.add_argument('--n_epochs', type=int, default=10, 
                        help='Number of epochs to evade')
    parser.add_argument('--defense_k', type=int, default=5,
                        help='Defense drops top-k per class per epoch')
    parser.add_argument('--true_label', type=int, default=0,
                        help='True label of base canary (will be misclassified away from this)')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epsilon', type=float, default=10.0)
    parser.add_argument('--delta', type=float, default=1e-5)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--step_size', type=float, default=0.01,
                        help='Perturbation step size')
    parser.add_argument('--n_iterations', type=int, default=20,
                        help='Perturbation iterations per epoch')
    parser.add_argument('--warmup_epochs', type=int, default=2,
                        help='Number of epochs before first drop to start perturbations')
    parser.add_argument('--output', type=str, default='defense_aware_canary.pt',
                        help='Output file for canary')
    parser.add_argument('--seed', type=int, default=0)
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load data
    print(f"Loading {args.data_name} dataset...")
    X_train, y_train, out_dim = load_data(args.data_name, n_df=None, split='train')
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize model to get fixed init weights
    in_shape = X_train.shape
    init_model = Models[args.model_name](in_shape, out_dim=out_dim)
    init_state = deepcopy(init_model.state_dict())
    
    # Start with a random sample from the true label class as base canary
    true_label_mask = y_train == args.true_label
    true_label_indices = torch.where(true_label_mask)[0]
    base_canary_idx = true_label_indices[0].item()  # Take first sample of that class
    base_canary = X_train[base_canary_idx].clone()
    
    print(f"Base canary: sample {base_canary_idx} from class {args.true_label}")
    
    # Craft defense-aware canary
    canary, final_label = craft_defense_aware_canary(
        base_canary=base_canary,
        true_label=args.true_label,
        X_train=X_train,
        y_train=y_train,
        model_name=args.model_name,
        init_model=init_state,
        n_epochs=args.n_epochs,
        defense_k=args.defense_k,
        batch_size=args.batch_size,
        lr=args.lr,
        epsilon=args.epsilon,
        delta=args.delta,
        max_grad_norm=args.max_grad_norm,
        perturbation_step_size=args.step_size,
        perturbation_iterations=args.n_iterations,
        warmup_epochs=args.warmup_epochs,
        device=device,
        verbose=True
    )
    
    # Save canary
    torch.save({
        'canary': canary,
        'true_label': args.true_label,
        'audit_label': final_label,  # Use this label for auditing!
        'init_model': init_state,
        'args': vars(args)
    }, args.output)
    
    print(f"\nCanary saved to {args.output}")
    print(f"  True label: {args.true_label}")
    print(f"  Audit label (use this for auditing): {final_label}")
