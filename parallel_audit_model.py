"""Auditing DP-SGD in black-box setting - Modified for model parallelism"""
import os
import sys
import time
import copy
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from tqdm import tqdm
import numpy as np
import argparse
from opacus.accountants.utils import get_noise_multiplier
from torch.utils.data import TensorDataset, DataLoader, Dataset
import dill
import pdb
import matplotlib.pyplot as plt

from models import Models
from models.wideresnet import WSConv2d
from utils.data import load_data
from utils.dpsgd import clip_and_accum_grads, get_per_sample_grads
from utils.audit import compute_eps_lower_from_mia, compute_eps_lower_from_mia_given_t
from utils.clipbkd import craft_clipbkd, choose_worstcase_label

import gc
import torch.nn.functional as F
import torchvision.transforms.v2 as v2

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

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


def cleanup():
    """Cleanup - no distributed operations needed"""
    pass


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
               gradient_space_audit=False, crafted_gradient=None, defense=False, device='cuda:0'):
    """
    Train a single model on a single GPU (no DDP).
    """
    # Move everything to the specified device
    device = torch.device(device)
    
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

    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # Set DP noise
    if epsilon is not None:
        noise_multiplier = get_noise_multiplier(
            target_epsilon=epsilon,
            target_delta=delta,
            sample_rate=batch_size / len(X),
            epochs=n_epochs,
            accountant='rdp'
        )
    else:
        noise_multiplier = 0

    assert block_size <= batch_size, "block_size must be smaller than batch_size"

    if len(X.shape) > 2:
        aug_fn = AugmentationFunction(X.shape[2], X.shape[1])
    else:
        aug_fn = None

    # Create Dataset + DataLoader (no DDP sampler)
    dataset = IndexedTensorDataset(X, y)
    scores = np.zeros(len(dataset))
    drop_mask = np.zeros(len(dataset), dtype=np.int32)
    
    sampler = torch.utils.data.RandomSampler(
        dataset,
        replacement=False,
        num_samples=None,
        generator=torch.Generator().manual_seed(0)
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(0)
    )
    
    for epoch in range(n_epochs):
        epoch_start = time.time()
        optimizer.zero_grad()
        print(f"Epoch: {epoch} (Active samples: {int((drop_mask == 0).sum())}/{len(drop_mask)})", end='', flush=True)

        for batch_idx, (curr_X, curr_y, global_indices) in enumerate(loader):
            curr_X, curr_y = curr_X.to(device, non_blocking=True), curr_y.to(device, non_blocking=True)
            global_indices = global_indices.to(device, non_blocking=True)
            
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
                crafted_gradient=crafted_gradient
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
                        print("Adding noise:", noise_multiplier, max_grad_norm)
                        noise = noise_multiplier * max_grad_norm * torch.randn_like(grad)
                        grad.add_(noise)
                    
                    if param.grad is None:
                        param.grad = grad.clone()
                    else:
                        param.grad.copy_(grad)
            
            optimizer.step()
            optimizer.zero_grad()
        
        epoch_time = time.time() - epoch_start
        print(f" | Time: {epoch_time:.2f}s")
        
        # Defense operations
        if defense:
            print('Defense')
            k = 5
            unique_classes = torch.unique(y).cpu()
            
            for cls in unique_classes:
                cls_indices = (y.cpu() == cls.item()).nonzero(as_tuple=True)[0]
                if len(cls_indices) == 0:
                    continue
                    
                cls_scores = torch.tensor(scores[cls_indices.cpu().numpy()], device=y.device)
                _, topk_indices = torch.topk(cls_scores, min(k, len(cls_scores)))
                topk_global_indices = cls_indices[topk_indices]
                
                dropped_indices = topk_global_indices.cpu().numpy()
                drop_mask[dropped_indices] = 1
                
                if X.shape[0] - 1 in dropped_indices and not hasattr(train_model, '_canary_dropped'):
                    print(f"\n[INFO] Canary (index {X.shape[0]-1}) was dropped from the training set!", drop_mask[-1])
                    train_model._canary_dropped = True
        
            scores.fill(0)

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
        np.save(f'{out_folder}/all_losses_in{suffix}.npy', all_losses['in'])
        np.save(f'{out_folder}/all_losses_out{suffix}.npy', all_losses['out'])


def resume_checkpoint(out_folder, fit_world_only, resume, rank=0):
    """Load checkpoint if resume is set to True and previous checkpoint exists"""
    outputs = {'out': [], 'in': []}
    losses = {'out': [], 'in': []}
    all_losses = {'in': [], 'out': []}
    train_set_accs = []
    test_set_accs = []

    suffix = f'_rank{rank}' if rank > 0 else ''

    if os.path.exists(out_folder) and resume:
        try:
            random_state = dill.load(open(f'{out_folder}/random_state{suffix}.dill', 'rb'))
            np.random.set_state(random_state['np'])
            torch.random.set_rng_state(random_state['torch'])

            if fit_world_only:
                outputs[fit_world_only] = np.load(f'{out_folder}/outputs_{fit_world_only}{suffix}.npy').tolist()
                losses[fit_world_only] = np.load(f'{out_folder}/losses_{fit_world_only}{suffix}.npy').tolist()
                all_losses[fit_world_only] = np.load(f'{out_folder}/all_losses_{fit_world_only}{suffix}.npy').tolist()

                if fit_world_only == 'out':
                    train_set_accs = np.load(f'{out_folder}/train_set_accs{suffix}.npy').tolist()
                    test_set_accs = np.load(f'{out_folder}/test_set_accs{suffix}.npy').tolist()
            else:
                outputs['in'] = np.load(f'{out_folder}/outputs_in{suffix}.npy').tolist()
                outputs['out'] = np.load(f'{out_folder}/outputs_out{suffix}.npy').tolist()
                train_set_accs = np.load(f'{out_folder}/train_set_accs{suffix}.npy').tolist()
                test_set_accs = np.load(f'{out_folder}/test_set_accs{suffix}.npy').tolist()
                losses['in'] = np.load(f'{out_folder}/losses_in{suffix}.npy').tolist()
                losses['out'] = np.load(f'{out_folder}/losses_out{suffix}.npy').tolist()
                all_losses['in'] = np.load(f'{out_folder}/all_losses_in{suffix}.npy').tolist()
                all_losses['out'] = np.load(f'{out_folder}/all_losses_out{suffix}.npy').tolist()
        except FileNotFoundError:
            print(f"[Rank {rank}] No checkpoint found, starting fresh")
    else:
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
    
    # # Get rank info from environment (but won't use NCCL/distributed ops)
    # local_rank = int(os.environ.get('LOCAL_RANK', 0))
    # rank = int(os.environ.get('RANK', 0))
    # world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    # print(f'[Rank {rank}] Starting with world_size={world_size}, local_rank={local_rank}')
    
    # # Set device for this process - NO distributed initialization
    # if torch.cuda.is_available():
    #     device = torch.device(f'cuda:{local_rank}')
    #     torch.cuda.set_device(device)
    #     print(f'[Rank {rank}] Using device: {torch.cuda.get_device_name(local_rank)}')
    # else:
    #     device = torch.device('cpu')
    #     print(f'[Rank {rank}] CUDA not available, using CPU')



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
    
    # Parse arguments
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--data_name', type=str, default='mnist')
    parser.add_argument('--model_name', type=str, default='lr', choices=list(Models.keys()))
    parser.add_argument('--n_reps', type=int, default=200)
    parser.add_argument('--n_df', type=int, default=0)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max_grad_norm', type=float, default=1)
    parser.add_argument('--epsilon', type=float, default=None)
    parser.add_argument('--delta', type=float, default=1e-5)
    parser.add_argument('--target_type', type=str, default='blank')
    parser.add_argument('--blank_alpha', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', type=str, default='exp_data/')
    parser.add_argument('--fixed_init', type=str, nargs='?', default=None, const='')
    parser.add_argument('--block_size', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--fit_world_only', type=str, default=None, choices=['in', 'out'])
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--badnets_label', type=int, default=-1)
    parser.add_argument('--view_badnets', action='store_true')
    parser.add_argument('--store_canary_rank', action='store_true')
    parser.add_argument('--holdout_audit', action='store_true')
    parser.add_argument('--target_class', type=int, default=0)
    parser.add_argument('--defense', action='store_true')
    parser.add_argument('--aug_mult', type=int, default=1)

    args = parser.parse_args()
    if args.max_grad_norm == -1:
        args.max_grad_norm = None

    # Reproducibility
    np.random.seed(args.seed + rank)  # Different seed per rank
    torch.manual_seed(args.seed + rank)

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
    
    if args.target_type == 'gradient_space_canary':
        target_X = X_out[-1].unsqueeze(0)
        target_y = y_out[-1].unsqueeze(0)
        if rank == 0:
            print("Using gradient-space canary")
    elif args.target_type == 'blank':
        blank_img = torch.zeros_like(X_out[[0]])
        if args.blank_alpha > 0:
            label_9_indices = (y_out == 9).nonzero(as_tuple=True)[0]
            if len(label_9_indices) > 0:
                label_9_img = X_out[label_9_indices[0]].unsqueeze(0)
                target_X = (1 - args.blank_alpha) * blank_img + args.blank_alpha * label_9_img
            else:
                target_X = blank_img
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
        # FGSM attack code (abbreviated for space)
        target_X = X_out[-1].unsqueeze(0)
        target_y = y_out[-1].unsqueeze(0)
    else:
        raise Exception(f'Target {args.target_type} not found')

    # Define datasets
    X_in, y_in = torch.vstack((X_out[:-1], target_X)), torch.cat((y_out[:-1], target_y))
    X_test, y_test, _ = load_data(args.data_name, None, split='test')
    
    if rank == 0:
        print('Training models')
    
    # Resume checkpoint - each rank loads its own
    worlds = [args.fit_world_only] if args.fit_world_only else ['in', 'out']
    
    outputs, losses, all_losses, train_set_accs, test_set_accs = resume_checkpoint(
        out_folder, args.fit_world_only, args.resume, rank)
    
    # Create crafted gradient if needed
    crafted_grad = None
    if args.target_type == 'gradient_space_canary':
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
                aug_mult=args.aug_mult,
                gradient_space_audit=args.target_type == 'gradient_space_canary' and world == 'in',
                crafted_gradient=crafted_grad if args.target_type == 'gradient_space_canary' and world == 'in' else None,
                device=device
            )
            
            # Compute outputs and losses
            model.eval()
            with torch.no_grad():
                target_X_device = target_X.to(device)
                target_y_device = target_y.to(device)
                
                output = model(target_X_device)
                
                if args.target_type == 'gradient_space_canary' and world == 'in' and crafted_grad is not None:
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
            
            # Each rank saves its own checkpoint
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

    dist.barrier()

    # Final audit - only rank 0 needs to combine results from all ranks
    if rank == 0:
        print("\n[Rank 0] Combining results from all ranks...")
        
        # Load results from all rank files
        combined_outputs = {'in': [], 'out': []}
        combined_losses = {'in': [], 'out': []}
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
                    if os.path.exists(f'{out_folder}/train_set_accs{suffix}.npy'):
                        combined_train_accs.extend(np.load(f'{out_folder}/train_set_accs{suffix}.npy'))
                    if os.path.exists(f'{out_folder}/test_set_accs{suffix}.npy'):
                        combined_test_accs.extend(np.load(f'{out_folder}/test_set_accs{suffix}.npy'))
                else:
                    combined_outputs[args.fit_world_only].extend(np.load(f'{out_folder}/outputs_{args.fit_world_only}{suffix}.npy'))
                    combined_losses[args.fit_world_only].extend(np.load(f'{out_folder}/losses_{args.fit_world_only}{suffix}.npy'))
            except FileNotFoundError:
                print(f"Warning: Could not find results for rank {r}")
        
        # Save combined results
        if not args.fit_world_only:
            np.save(f'{out_folder}/outputs_in.npy', combined_outputs['in'])
            np.save(f'{out_folder}/outputs_out.npy', combined_outputs['out'])
            np.save(f'{out_folder}/losses_in.npy', combined_losses['in'])
            np.save(f'{out_folder}/losses_out.npy', combined_losses['out'])
            if combined_train_accs:
                np.save(f'{out_folder}/train_set_accs.npy', combined_train_accs)
            if combined_test_accs:
                np.save(f'{out_folder}/test_set_accs.npy', combined_test_accs)
        
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