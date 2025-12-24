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
from utils.dpsgd import clip_and_accum_grads, DefenseConfig

import time
import torchvision.transforms.v2 as v2
from torch.utils.data import Dataset

# Enable performance optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def setup_device():
    """Return the torch device used for training/evaluation."""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    return device


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
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)
    
    for batch_X, batch_y, indices in loader:
        batch_X = batch_X.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)
        
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


class AugmentationFunction:
    def __init__(self, image_size=32, channels=3):
        self.base_transforms = v2.Compose([
            v2.RandomCrop(image_size, padding=4),
            v2.RandomHorizontalFlip(p=0.5),
        ])
    
    def __call__(self, x):
        return self.base_transforms(x)


class IndexedTensorDataset(Dataset):
    """A dataset that includes the index of each sample."""
    def __init__(self, *tensors):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        
    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors) + (index,)
        
    def __len__(self):
        return self.tensors[0].size(0)


def xavier_init_model(model):
    """Initialize model using Xavier initialization"""
    def init_weights(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)
    model.apply(init_weights)


def train_model_with_defense(model_name, X, y, epsilon, delta, max_grad_norm, 
                             n_epochs, lr, batch_size, init_model=None, out_dim=10, 
                             aug_mult=1, defense=False, defense_k=5, device='cuda:0', 
                             defense_score_norm='linf', defense_score_fn='grad_norm',
                             stop_on_canary_drop=False):
    """
    Train a model with the defense-aware training loop from parallel_audit_model.py.
    This allows tracking when the canary is dropped.
    
    Returns:
        model: Trained model
        canary_drop_epoch: Epoch when canary (last sample) was dropped, or -1 if never dropped
    """
    device = torch.device(device)
    torch.cuda.set_device(device)
    
    if init_model is None:
        model = Models[model_name](X.shape, out_dim=out_dim).to(device)
        if model_name == 'cnn':
            xavier_init_model(model)
    else:
        model = deepcopy(init_model).to(device)
    
    model.train()
    criterion = nn.CrossEntropyLoss()
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
        print(f"DP config: eps={epsilon}, delta={delta}, sample_rate={sample_rate:.6f}, epochs={n_epochs}, noise_multiplier={noise_multiplier}")
    else:
        noise_multiplier = 0
    
    block_size = min(batch_size, batch_size)
    
    if len(X.shape) > 2:
        aug_fn = AugmentationFunction(X.shape[2], X.shape[1])
    else:
        aug_fn = None
    
    # Create Dataset + DataLoader
    dataset = IndexedTensorDataset(X, y)
    scores = np.zeros(len(dataset))
    drop_mask = np.zeros(len(dataset), dtype=bool)
    
    sampler = torch.utils.data.RandomSampler(
        dataset,
        replacement=False,
        num_samples=None
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        pin_memory=True,
        num_workers=0,
        drop_last=False
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
    canary_drop_epoch = -1  # Track when canary is dropped
    
    for epoch in range(n_epochs):
        epoch_start = time.time()
        optimizer.zero_grad()
        print(f"Epoch: {epoch} (Active samples: {int((drop_mask == 0).sum())}/{len(drop_mask)})", end='', flush=True)
        
        for batch_idx, (curr_X, curr_y, global_indices) in enumerate(loader):
            curr_X, curr_y = curr_X.to(device, non_blocking=True), curr_y.to(device, non_blocking=True)
            global_indices = global_indices.to(device, non_blocking=True)
            
            if defense_score_fn == 'loss_momentum' and prev_losses is None:
                prev_losses = np.full((len(dataset),), np.nan, dtype=np.float32)
            
            if defense_score_fn == 'cos_update' and prev_params is None:
                prev_params = {n: p.detach().clone() for n, p in model.named_parameters()}
            
            if defense_score_fn == 'cos_theta0' and theta0_params is None:
                theta0_params = {n: p.detach().clone() for n, p in model.named_parameters()}
            
            if defense_score_fn == 'cos_theta0' and theta0_params is not None:
                curr_params = {n: p.detach() for n, p in model.named_parameters()}
                theta_t_minus_theta0 = torch.cat([(curr_params[n] - theta0_params[n]).reshape(-1) for n in theta0_params.keys()], dim=0)
            else:
                theta_t_minus_theta0 = None
            
            defense_cfg = DefenseConfig(
                score_fn=defense_score_fn,
                score_norm=defense_score_norm,
                delta_theta=prev_delta_theta,
                theta_t_minus_theta0=theta_t_minus_theta0,
                prev_losses=prev_losses,
                loss_hist=loss_hist,
                loss_hist_pos=loss_hist_pos,
                loss_volatility_k=5,
                grad_norm_hist=grad_norm_hist,
                grad_norm_hist_pos=grad_norm_hist_pos,
                grad_norm_percentile_k=20,
                grad_dir_hist=grad_dir_hist,
                grad_dir_hist_pos=grad_dir_hist_pos,
                grad_dir_volatility_k=5,
                grad_dir_proj=grad_dir_proj,
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
            
            # Clip & accumulate gradients
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
                world_size=1,
                rank=0,
                batch_size=batch_size,
                is_gradient_space_canary=False,
                crafted_gradient=None,
                defense_cfg=defense_cfg,
                defense_apply_ascent=True
            )
            
            drop_mask[drop_mask == 1] = 2
            
            # Apply the accumulated gradients
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name not in curr_accumulated_gradients:
                        print(f"Warning: Parameter {name} not found in accumulated gradients")
                        continue
                    
                    grad = curr_accumulated_gradients[name].to(device)
                    
                    # Add DP noise if needed
                    if noise_multiplier > 0 and max_grad_norm is not None:
                        batch_size_in = int(curr_X.shape[0])
                        noise_std = noise_multiplier * max_grad_norm / float(batch_size_in)
                        noise = noise_std * torch.randn_like(grad)
                        grad.add_(noise)
                    
                    if param.grad is None:
                        param.grad = grad.clone()
                    else:
                        param.grad.copy_(grad)
            
            optimizer.step()
            optimizer.zero_grad()
            
            if defense_score_fn == 'cos_update' and prev_params is not None:
                curr_params = {n: p.detach() for n, p in model.named_parameters()}
                prev_delta_theta = torch.cat([(curr_params[n] - prev_params[n]).reshape(-1) for n in prev_params.keys()], dim=0)
                prev_params = {n: curr_params[n].clone() for n in prev_params.keys()}
        
        epoch_time = time.time() - epoch_start
        print(f" | Time: {epoch_time:.2f}s")
        
        # Defense operations
        if defense:
            unique_classes = torch.unique(y).cpu()
            active_mask = torch.from_numpy(drop_mask == 0)
            
            for cls in unique_classes:
                cls_indices = ((y.cpu() == cls.item()) & active_mask).nonzero(as_tuple=True)[0]
                if len(cls_indices) == 0:
                    continue
                
                cls_scores = torch.tensor(scores[cls_indices.cpu().numpy()], device=y.device)
                _, topk_indices = torch.topk(cls_scores, min(defense_k, len(cls_scores)))
                
                topk_global_indices = cls_indices[topk_indices]
                
                dropped_indices = topk_global_indices.cpu().numpy()
                drop_mask[dropped_indices] = 1
                
                if X.shape[0] - 1 in dropped_indices:
                    print(f"\n[INFO] Canary (index {X.shape[0]-1}) was dropped from the training set at epoch {epoch}!")
                    if canary_drop_epoch == -1:  # Record first drop
                        canary_drop_epoch = epoch
            
            if stop_on_canary_drop and canary_drop_epoch != -1:
                break
            
            scores.fill(0)
    
    return model, canary_drop_epoch


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


def perturb_canary_unified(
    canary,
    true_label,
    model_state_dict,
    model_name,
    in_shape,
    out_dim,
    device,
    n_iterations=20,
    step_size=0.01,
    drop_threshold=0.0,
    tau_drop=0.1,
    lambda_drop=1.0,
    verbose=False,
):
    canary = canary.clone().detach().to(device)
    true_label_tensor = torch.tensor([true_label]).to(device)
    criterion = nn.CrossEntropyLoss()

    last_pred = true_label

    for iteration in range(n_iterations):
        fresh_model = Models[model_name](in_shape, out_dim=out_dim)
        fresh_model = make_opacus_compatible(fresh_model)
        fresh_model.load_state_dict(model_state_dict, strict=False)
        fresh_model.to(device)
        fresh_model.train()

        canary.requires_grad = True

        output = fresh_model(canary.unsqueeze(0))
        loss_true = criterion(output, true_label_tensor)

        pred = output.argmax(dim=1).item()
        last_pred = pred

        params = [p for p in fresh_model.parameters() if p.requires_grad]
        grads = torch.autograd.grad(
            loss_true,
            params,
            create_graph=True,
            retain_graph=True,
            allow_unused=False,
        )
        grad_norm_sq = torch.zeros((), device=device)
        for g in grads:
            grad_norm_sq = grad_norm_sq + g.pow(2).sum()
        grad_norm = torch.sqrt(grad_norm_sq + 1e-12)

        tau = float(tau_drop)
        if tau <= 0:
            raise ValueError(f"tau_drop must be > 0, got {tau}")
        p_dropped = torch.sigmoid((grad_norm - float(drop_threshold)) / tau)

        meta_obj = loss_true - float(lambda_drop) * p_dropped
        grad_x = torch.autograd.grad(meta_obj, canary, retain_graph=False)[0]

        if pred != int(true_label) and grad_norm.item() < float(drop_threshold):
            del fresh_model
            break

        if verbose and iteration % 5 == 0:
            print(
                f"    Iter {iteration}: loss_true={loss_true.item():.4f}, "
                f"grad_norm={grad_norm.item():.4f}, p_drop={p_dropped.item():.4f}, meta={meta_obj.item():.4f}, "
                f"pred={pred}, true={true_label}, thr={float(drop_threshold):.4f}"
            )

        with torch.no_grad():
            canary = (canary + step_size * grad_x.sign()).detach()

        del fresh_model

    return canary.detach().cpu(), last_pred




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
    max_iter=20,
    perturbation_step_size=0.01,
    perturbation_iterations=20,
    lambda_drop=1.0,
    tau_drop=0.1,
    device='cuda',
    verbose=True
):
    """
    Craft a canary that evades the top-k gradient norm defense for all epochs.
    Uses untargeted attack (gradient ascent) to cause misclassification.
    
    Simplified; trains fresh model each epoch.
    
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
        device: Device to use
        verbose: Print progress
    
    Returns:
        Tuple of (perturbed canary, final misclassified label to use for auditing)
    """
    canary = base_canary.clone().detach()
    
    in_shape = X_train.shape
    out_dim = len(torch.unique(y_train))
    canary_idx = len(X_train)  # Canary will be appended at the end
    
    canary_label = true_label
    audit_label = true_label
    
    if verbose:
        print(f"Crafting defense-aware canary (untargeted + defense evasion)")
        print(f"  Model: {model_name}")
        print(f"  Epochs to evade: {n_epochs}")
        print(f"  Defense k: {defense_k}")
        print(f"  True label: {true_label}")
        print(f"  Max canary update iterations: {max_iter}")
        print(f"  Training set size: {len(X_train)}")
        print(f"  Privacy: ε={epsilon}, δ={delta}")

    for it in range(int(max_iter)):
        if verbose:
            print(f"\n=== Canary update iter {it}/{int(max_iter) - 1} ===")

        canary_label_tensor = torch.tensor([canary_label])
        X_with_canary = torch.cat([X_train, canary.unsqueeze(0)], dim=0)
        y_with_canary = torch.cat([y_train, canary_label_tensor], dim=0)

        model = Models[model_name](in_shape, out_dim=out_dim)
        if init_model is not None:
            model.load_state_dict(deepcopy(init_model))

        if verbose:
            print(f"  Training model for full {n_epochs} epochs with defense...")

        model, drop_epoch = train_model_with_defense(
            model_name=model_name,
            X=X_with_canary,
            y=y_with_canary,
            epsilon=epsilon,
            delta=delta,
            max_grad_norm=max_grad_norm,
            n_epochs=n_epochs,
            lr=lr,
            batch_size=batch_size,
            init_model=model,
            out_dim=out_dim,
            aug_mult=1,
            defense=True,
            defense_k=defense_k,
            device=device,
            stop_on_canary_drop=True,
        )

        model.to(device)
        with torch.no_grad():
            final_pred = model(canary.unsqueeze(0).to(device)).argmax(dim=1).item()

        audit_label = final_pred

        success = (final_pred != true_label) and (drop_epoch == -1)

        if verbose:
            print(f"  Final pred on canary: {final_pred} (true={true_label})")
            print(f"  Canary ever dropped: {drop_epoch != -1}")

        if success:
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            break

        model_state = model.state_dict()
        grad_norms = compute_per_sample_gradient_norms(
            model_state, model_name, in_shape, out_dim,
            X_with_canary, y_with_canary, device, batch_size
        )
        thresholds = get_top_k_threshold_per_class(
            grad_norms, y_with_canary.numpy(), defense_k
        )
        drop_threshold = thresholds[canary_label]

        if verbose:
            print(f"  Updating canary with untargeted + drop-penalty objective...")

        canary, _ = perturb_canary_unified(
            canary,
            true_label,
            model_state,
            model_name,
            in_shape,
            out_dim,
            device,
            n_iterations=perturbation_iterations,
            step_size=perturbation_step_size,
            drop_threshold=drop_threshold,
            tau_drop=tau_drop,
            lambda_drop=lambda_drop,
            verbose=verbose,
        )

        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    if verbose:
        print(f"\n=== Final Result ===")
        print(f"  Final canary label for auditing: {audit_label}")
        print(f"  True label: {true_label}")
    
    return canary, audit_label


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
    parser.add_argument('--use_last_sample_as_canary', action='store_true',
                        help='Use the last sample in the training set as the canary base and remove it from training (mislabel-last)')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epsilon', type=float, default=10.0)
    parser.add_argument('--delta', type=float, default=1e-5)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--step_size', type=float, default=0.01,
                        help='Perturbation step size')
    parser.add_argument('--n_iterations', type=int, default=20,
                        help='Perturbation iterations per epoch')
    parser.add_argument('--max_iter', type=int, default=20,
                        help='Max number of canary update iterations (each trains a fresh model for full n_epochs)')
    parser.add_argument('--lambda_drop', type=float, default=1.0,
                        help='Weight on drop-penalty term in unified canary objective')
    parser.add_argument('--tau_drop', type=float, default=0.1,
                        help='Temperature for sigmoid approximation of binary dropped indicator')
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
    
    device = setup_device()
    print(f"Using device: {device}")
    
    # Initialize model to get fixed init weights
    in_shape = X_train.shape
    init_model = Models[args.model_name](in_shape, out_dim=out_dim)
    init_state = deepcopy(init_model.state_dict())
    
    # Start with a random sample from the true label class as base canary
    if args.use_last_sample_as_canary:
        base_canary_idx = int(len(X_train) - 1)
        base_canary = X_train[base_canary_idx].clone()
        args.true_label = int(y_train[base_canary_idx].item())
        X_train = X_train[:-1].clone()
        y_train = y_train[:-1].clone()
        print(f"Base canary: last sample (index {base_canary_idx}) removed from training; true_label={args.true_label}")
    else:
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
        max_iter=args.max_iter,
        perturbation_step_size=args.step_size,
        perturbation_iterations=args.n_iterations,
        lambda_drop=args.lambda_drop,
        tau_drop=args.tau_drop,
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
    
    # Final cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
