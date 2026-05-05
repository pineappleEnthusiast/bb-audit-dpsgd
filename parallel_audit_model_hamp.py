"""
Label-only MIA audit with HAMP and filtering defenses.

Trains shadow models and audits privacy using a label-only attack that:
1. Queries the model on 18 augmentations of the canary
2. Records binary correctness vector
3. Fits logistic regression (LOO CV) to score membership

Supports three defense types:
- none: standard CE training
- hamp: entropy regularization + label smoothing (train-time) + confidence randomization (test-time)
- filter: gradient-norm filtering during training
"""

import os
import time
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.optim as optim
import numpy as np
import argparse
from pathlib import Path
from torch.utils.data import DataLoader

from models import Models
from utils.data import load_data
from utils.training import (
    xavier_init_model, init_wideresnet, IndexedTensorDataset
)
from utils.dpsgd import clip_and_accum_grads, DefenseConfig
from utils.audit import compute_eps_lower_from_mia
from utils.args import build_parser
from sklearn.linear_model import LogisticRegression

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


# ============================================================================
# HAMP Helper Functions
# ============================================================================

def compute_p_from_target_entropy(gamma, num_classes):
    """
    Binary search for p such that entropy of [p, q, q, ..., q] equals gamma * log(C).

    Args:
        gamma: target entropy as fraction of max entropy (0-1)
        num_classes: number of classes C

    Returns:
        p: probability for correct class
    """
    target_entropy = gamma * np.log(num_classes)

    def entropy(p, C):
        if p <= 0 or p > 1:
            return 0
        q = (1 - p) / (C - 1)
        if q < 0:
            return 0
        h = -p * np.log(p + 1e-10)
        if C > 1 and q > 0:
            h -= (C - 1) * q * np.log(q + 1e-10)
        return h

    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        h = entropy(mid, num_classes)
        if h < target_entropy:
            hi = mid
        else:
            lo = mid
    return lo


def generate_soft_labels(y, num_classes, p):
    """
    Generate soft labels for HAMP training.

    For each sample, set true class to p and others to (1-p)/(C-1).

    Args:
        y: class indices, shape (B,)
        num_classes: C
        p: probability for true class

    Returns:
        soft labels, shape (B, C)
    """
    B = len(y)
    soft = torch.full((B, num_classes), (1 - p) / (num_classes - 1), dtype=torch.float32)
    soft[torch.arange(B), y] = p
    return soft


def kl_divergence_with_entropy_regularization(logits, soft_labels, alpha_entropy):
    """
    HAMP training loss: KL(soft_labels || softmax(logits)) - alpha * H(softmax(logits)).

    The entropy term is subtracted (maximized) to encourage high-entropy predictions.
    """
    probs = F.softmax(logits, dim=1)

    # KL divergence
    kl_loss = F.kl_div(
        F.log_softmax(logits, dim=1),
        soft_labels,
        reduction='batchmean'
    )

    # Entropy regularization (negative entropy since we subtract)
    entropy = -(probs * F.log_softmax(logits, dim=1)).sum(dim=1).mean()

    return kl_loss - alpha_entropy * entropy


def rank_preserving_score_replacement(logits):
    """
    HAMP test-time defense: randomize confidence while preserving predicted class order.

    For each sample:
    1. Get rank order of logits
    2. Draw random values and sort them in the same order
    3. Scatter back to get randomized logits

    The argmax (top-1 prediction) is preserved; confidence is randomized.
    """
    B, C = logits.shape

    # Get rank order
    _, rank_indices = torch.sort(logits, dim=1)

    # Draw and sort random values
    random_vals = torch.rand(B, C, device=logits.device, dtype=logits.dtype)
    random_sorted = torch.argsort(random_vals, dim=1)

    # Scatter back preserving rank order
    result = torch.zeros_like(logits)
    for i in range(B):
        result[i, rank_indices[i]] = random_vals[i, random_sorted[i]]

    return result


# ============================================================================
# Augmentation and Attack Functions
# ============================================================================

def generate_augmentations(x, num_augmentations, use_flip=True):
    """
    Generate fixed augmentations: shifts + horizontal flips.

    For CIFAR (use_flip=True): 2 flips × 3 shifts × 3 shifts = 18 augmentations
    For MNIST (use_flip=False): first 18 of 5×5 shifts

    Args:
        x: input tensor, shape (..., H, W) or (..., C, H, W)
        num_augmentations: number to generate
        use_flip: whether to apply horizontal flips

    Returns:
        list of augmented tensors
    """
    augmentations = []
    shifts = [-4, -2, 0] if use_flip else list(range(-2, 3))

    flip_opts = [False, True] if use_flip else [False]

    for do_flip in flip_opts:
        for shift_h in shifts:
            for shift_w in shifts:
                if len(augmentations) >= num_augmentations:
                    return augmentations

                aug = x.clone()
                if do_flip:
                    aug = torch.flip(aug, dims=[-2])

                aug = torch.roll(aug, shifts=shift_h, dims=-2)
                aug = torch.roll(aug, shifts=shift_w, dims=-1)

                augmentations.append(aug)

    return augmentations


def generate_binary_correctness_vector(model, x, y, augmentations, device):
    """
    Label-only attack: query model on augmentations and record correctness.

    For each augmentation, returns 1 if argmax == y, else 0.
    This binary vector is the attack signal.

    Args:
        model: trained model (in eval mode)
        x: original input (unused, augmentations already prepared)
        y: true class
        augmentations: list of augmented inputs
        device: torch device

    Returns:
        binary vector, shape (len(augmentations),) as numpy float32
    """
    model.eval()
    binary_vector = []

    with torch.no_grad():
        for aug in augmentations:
            aug = aug.to(device)
            logits = model(aug.unsqueeze(0))
            pred = logits.argmax(dim=1).item()
            binary_vector.append(1.0 if pred == y else 0.0)

    return np.array(binary_vector, dtype=np.float32)


# ============================================================================
# Training Functions
# ============================================================================

def train_model(model, X, y, canary_x, canary_y, device, args, defense_type='none'):
    """
    Train a model with optional HAMP or gradient-norm filter defense.

    Args:
        model: PyTorch model
        X, y: training data and labels (numpy), already includes canary if in-world
        canary_x, canary_y: unused here, kept for signature consistency
        device: torch device
        args: argument namespace
        defense_type: 'none', 'hamp', or 'filter'

    Returns:
        trained model
    """
    model.train()
    batch_size = args.batch_size if args.batch_size else 256
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y).long()
    dataset = IndexedTensorDataset(X_tensor, y_tensor)
    num_classes = int(y_tensor.max().item()) + 1

    if defense_type in ('none', 'hamp'):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        hamp_gamma = getattr(args, 'hamp_gamma', 0.95)
        hamp_alpha_entropy = getattr(args, 'hamp_alpha_entropy', 1.0)

        if defense_type == 'hamp':
            p = compute_p_from_target_entropy(hamp_gamma, num_classes)

        for epoch in range(args.n_epochs):
            for X_b, y_b, _ in loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                optimizer.zero_grad()
                logits = model(X_b)

                if defense_type == 'hamp':
                    soft_labels = generate_soft_labels(y_b, num_classes, p).to(device)
                    loss = kl_divergence_with_entropy_regularization(logits, soft_labels, hamp_alpha_entropy)
                else:
                    loss = F.cross_entropy(logits, y_b)

                loss.backward()
                optimizer.step()

    elif defense_type == 'filter':
        block_size = args.block_size if args.block_size else batch_size
        block_size = min(block_size, batch_size)
        max_grad_norm = args.max_grad_norm
        defense_k = args.defense_k
        defense_apply_ascent = args.defense_apply_ascent
        defense_score_fn = args.defense_score_fn
        defense_score_norm = args.defense_score_norm
        defense_filter_every = getattr(args, 'defense_filter_every', 1)

        criterion = nn.CrossEntropyLoss()
        scores = np.zeros(len(dataset), dtype=np.float32)
        # 0 = active, 1 = flagged for gradient ascent, 2 = dropped
        drop_mask = np.zeros(len(dataset), dtype=np.int8)

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Lazy projection matrix state (updated by clip_and_accum_grads_block internally)
        grad_dir_proj = rand_proj_mat = maxmin_proj_mat = None
        alignment_proj_mat = grad_accel_proj = grad_jerk_proj = None
        grad_norm_hist = grad_norm_hist_pos = None
        grad_dir_hist = grad_dir_hist_pos = None
        grad_accel_hist = grad_accel_hist_pos = None
        grad_jerk_hist = grad_jerk_hist_pos = None
        dir_unique_hist = dir_unique_hist_pos = None
        prev_losses = loss_hist = loss_hist_pos = None

        for epoch in range(args.n_epochs):
            optimizer.zero_grad()

            for curr_X, curr_y, global_indices in loader:
                curr_X = curr_X.to(device)
                curr_y = curr_y.to(device)
                global_indices = global_indices.to(device)

                defense_cfg = DefenseConfig(
                    score_fn=defense_score_fn,
                    score_norm=defense_score_norm,
                    grad_norm_hist=grad_norm_hist,
                    grad_norm_hist_pos=grad_norm_hist_pos,
                    grad_norm_percentile_k=args.grad_norm_percentile_k,
                    grad_dir_hist=grad_dir_hist,
                    grad_dir_hist_pos=grad_dir_hist_pos,
                    grad_dir_volatility_k=args.grad_dir_volatility_k,
                    grad_dir_proj=grad_dir_proj,
                    rand_proj_mat=rand_proj_mat,
                    rand_proj_var_m=args.rand_proj_var_m,
                    maxmin_proj_mat=maxmin_proj_mat,
                    maxmin_proj_k=args.maxmin_proj_k,
                    grad_rank_mode=args.grad_rank_mode,
                    grad_rank_eps=args.grad_rank_eps,
                    grad_accel_hist=grad_accel_hist,
                    grad_accel_hist_pos=grad_accel_hist_pos,
                    grad_accel_proj=grad_accel_proj,
                    grad_jerk_hist=grad_jerk_hist,
                    grad_jerk_hist_pos=grad_jerk_hist_pos,
                    grad_jerk_proj=grad_jerk_proj,
                    alignment_proj_mat=alignment_proj_mat,
                    alignment_proj_k=args.alignment_proj_k,
                    dir_unique_hist=dir_unique_hist,
                    dir_unique_hist_pos=dir_unique_hist_pos,
                    dir_unique_k=args.dir_unique_k,
                    grad_scatter_k=args.grad_scatter_k,
                    prev_losses=prev_losses,
                    loss_hist=loss_hist,
                    loss_hist_pos=loss_hist_pos,
                    loss_volatility_k=args.loss_volatility_k,
                )

                batch_indices = global_indices.cpu().numpy()
                batch_drop_mask = drop_mask[batch_indices]

                curr_accumulated_gradients, scores = clip_and_accum_grads(
                    model, curr_X, curr_y, optimizer, criterion, max_grad_norm,
                    drop_mask=batch_drop_mask,
                    block_size=block_size,
                    scores=scores,
                    device=device,
                    global_indices=global_indices,
                    world_size=1,
                    rank=0,
                    batch_size=batch_size,
                    defense_cfg=defense_cfg,
                    defense_apply_ascent=defense_apply_ascent,
                )

                # Propagate lazily created projection matrices back from defense_cfg
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

                # Samples that received gradient ascent (1) are now permanently dropped (2)
                drop_mask[batch_indices[batch_drop_mask == 1]] = 2

                # Apply the accumulated gradients
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if name not in curr_accumulated_gradients:
                            continue
                        grad = curr_accumulated_gradients[name].to(device)
                        grad.div_(float(batch_size))
                        if param.grad is None:
                            param.grad = grad.clone()
                        else:
                            param.grad.copy_(grad)

                optimizer.step()
                optimizer.zero_grad()

            # Filter top-k samples per class at end of each epoch
            if epoch % defense_filter_every == 0:
                active_mask = torch.from_numpy(drop_mask == 0)
                for cls in torch.unique(y_tensor):
                    cls_indices = (
                        (y_tensor == cls.item()) & active_mask
                    ).nonzero(as_tuple=True)[0]
                    if len(cls_indices) == 0:
                        continue
                    cls_scores = torch.tensor(scores[cls_indices.numpy()])
                    _, topk_idx = torch.topk(cls_scores, min(defense_k, len(cls_scores)))
                    drop_mask[cls_indices[topk_idx].numpy()] = 1
                scores.fill(0)

    return model


# ============================================================================
# Audit Function
# ============================================================================

def audit(binary_vectors_in, binary_vectors_out, alpha, delta):
    """
    Audit membership inference using LOO logistic regression.

    Args:
        binary_vectors_in: shape (N, 18) - binary vectors from in-world models
        binary_vectors_out: shape (N, 18) - binary vectors from out-world models
        alpha: significance level
        delta: DP delta parameter

    Returns:
        dict with 'emp_eps', 'scores', 'labels', 'threshold'
    """
    # Stack features: (2N, 18)
    all_features = np.vstack([binary_vectors_in, binary_vectors_out])
    # Stack labels: 1 for in, 0 for out
    all_labels = np.concatenate([
        np.ones(len(binary_vectors_in), dtype=np.int64),
        np.zeros(len(binary_vectors_out), dtype=np.int64)
    ])

    # LOO cross-validation with logistic regression
    scores = np.zeros(len(all_features))
    for i in range(len(all_features)):
        # Train on all except i-th
        train_mask = np.ones(len(all_features), dtype=bool)
        train_mask[i] = False

        X_train = all_features[train_mask]
        y_train = all_labels[train_mask]

        clf = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs')
        clf.fit(X_train, y_train)

        # Score held-out sample (probability of class 1)
        scores[i] = clf.predict_proba(all_features[i:i+1])[0, 1]

    # Compute empirical epsilon
    emp_eps = compute_eps_lower_from_mia(
        torch.from_numpy(scores).float(),
        torch.from_numpy(all_labels).long(),
        alpha,
        delta,
        method='GDP'
    )

    return {
        'emp_eps': float(emp_eps),
        'scores': scores,
        'labels': all_labels,
    }


# ============================================================================
# Utility Functions
# ============================================================================

def distribute_reps(n_reps, world_size):
    """Distribute reps across ranks (round-robin)."""
    reps_per_rank = [[] for _ in range(world_size)]
    for i in range(n_reps):
        reps_per_rank[i % world_size].append(i)
    return reps_per_rank


# ============================================================================
# Main
# ============================================================================

def main():
    # Parse arguments
    parser = build_parser()

    # Add HAMP-specific arguments
    parser.add_argument('--defense_type', type=str, default='none',
                        choices=['none', 'hamp', 'filter'],
                        help='Defense type: none, hamp, or filter')
    parser.add_argument('--hamp_gamma', type=float, default=0.95,
                        help='HAMP target entropy as fraction of max (0-1)')
    parser.add_argument('--hamp_alpha_entropy', type=float, default=1.0,
                        help='HAMP entropy regularization weight')

    args = parser.parse_args()

    # Distributed setup
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        dist.init_process_group(backend='nccl', init_method='env://')
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        rank = int(os.environ.get('RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        device = torch.device(f'cuda:{local_rank}')
    else:
        rank = 0
        world_size = 1
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    is_rank_zero = (rank == 0)

    # Seeding
    seed = args.seed
    np.random.seed(seed + rank)

    # Load data (load_data returns tensors; convert to numpy for this script)
    X, y, out_dim = load_data(args.data_name, n_df=args.n_df)
    X = X.numpy() if isinstance(X, torch.Tensor) else X
    y = y.numpy() if isinstance(y, torch.Tensor) else y

    # Load or create canary
    canary_x, canary_y = None, None
    if args.target_type == 'blank':
        canary_x = torch.zeros(X.shape[1:], dtype=torch.float32)
        canary_y = 0
    elif args.target_type == 'mislabeled':
        # Use a random training sample, mislabel it
        idx = np.random.randint(len(X))
        canary_x = torch.from_numpy(X[idx]).float()
        canary_y = args.mislabeled_target_class
    elif args.target_type.endswith('.pt'):
        canary_data = torch.load(args.target_type)
        canary_x = canary_data['canary'].float()
        canary_y = int(canary_data['label'])
    else:
        raise ValueError(f"Unknown target_type: {args.target_type}")

    if is_rank_zero:
        print(f"Data: {args.data_name}, shape={X.shape}, out_dim={out_dim}")
        print(f"Canary: {args.target_type}, label={canary_y}")
        print(f"Defense: {args.defense_type}")
        print(f"Distributed: rank={rank}, world_size={world_size}")

    # Fixed init model — all ranks use the same seed so they share identical initialization
    torch.manual_seed(seed)
    init_model = Models[args.model_name](X.shape, out_dim=out_dim)
    if args.model_name == 'wideresnet':
        init_wideresnet(init_model)
    else:
        xavier_init_model(init_model)
    # Rank-specific seed for per-rep training variation
    torch.manual_seed(seed + rank)

    # Distribute reps
    reps_per_rank = distribute_reps(args.n_reps // 2, world_size)
    my_reps = reps_per_rank[rank]

    binary_vectors = {'in': [], 'out': []}

    # Training loop
    for world in ['in', 'out']:
        for rep_id in my_reps:
            if is_rank_zero:
                print(f"[{world}] Rep {rep_id}")

            # Prepare data for this world
            X_world = X.copy()
            y_world = y.copy()

            # For in-world: include canary in training set
            if world == 'in':
                canary_x_np = canary_x.cpu().numpy() if isinstance(canary_x, torch.Tensor) else canary_x
                X_world = np.vstack([X_world, canary_x_np[np.newaxis, ...]])
                y_world = np.concatenate([y_world, [canary_y]])
            # For out-world: canary is NOT in training set

            # Create model copy
            model = copy.deepcopy(init_model)
            model.to(device)

            # Train
            train_model(model, X_world, y_world, canary_x, canary_y, device, args,
                       defense_type=args.defense_type)

            # Generate augmentations and get binary vector
            use_flip = (args.data_name != 'mnist')
            augmentations = generate_augmentations(canary_x, 18, use_flip=use_flip)
            binary_vec = generate_binary_correctness_vector(model, canary_x, canary_y,
                                                            augmentations, device)
            binary_vectors[world].append(binary_vec)

    # Save per-rank results
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(binary_vectors['in']) > 0:
        np.save(output_dir / f'binary_vectors_in_rank{rank}.npy',
                np.array(binary_vectors['in']))
        np.save(output_dir / f'binary_vectors_out_rank{rank}.npy',
                np.array(binary_vectors['out']))

    if world_size > 1:
        dist.barrier()

    # Rank 0: aggregate and audit
    if is_rank_zero:
        all_binary_in = []
        all_binary_out = []

        for r in range(world_size):
            path_in = output_dir / f'binary_vectors_in_rank{r}.npy'
            path_out = output_dir / f'binary_vectors_out_rank{r}.npy'

            if path_in.exists():
                all_binary_in.append(np.load(path_in))
            if path_out.exists():
                all_binary_out.append(np.load(path_out))

        if len(all_binary_in) > 0 and len(all_binary_out) > 0:
            binary_vectors_in = np.vstack(all_binary_in)
            binary_vectors_out = np.vstack(all_binary_out)

            # Run audit
            audit_result = audit(binary_vectors_in, binary_vectors_out,
                                alpha=args.alpha, delta=args.delta)

            # Save outputs
            np.save(output_dir / 'binary_vectors_in.npy', binary_vectors_in)
            np.save(output_dir / 'binary_vectors_out.npy', binary_vectors_out)
            np.save(output_dir / 'emp_eps.npy', np.array(audit_result['emp_eps']))
            np.save(output_dir / 'mia_scores.npy', audit_result['scores'])
            np.save(output_dir / 'mia_labels.npy', audit_result['labels'])

            print(f"Empirical epsilon: {audit_result['emp_eps']:.6f}")
            print(f"Outputs saved to {output_dir}")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
