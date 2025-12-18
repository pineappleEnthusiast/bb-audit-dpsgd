import argparse
import copy
import os
import time
import warnings

import dill
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from opacus import PrivacyEngine
from opacus.accountants.utils import get_noise_multiplier
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.validators import ModuleValidator
from torch.utils.data import DataLoader, Dataset, TensorDataset

import matplotlib.pyplot as plt
import torchvision.transforms.v2 as v2
from privacy_estimates import AttackResults
from sklearn.linear_model import LogisticRegression

from models import Models
from models.wideresnet import WSConv2d
from utils.audit import compute_eps_lower_from_mia, compute_eps_lower_from_mia_given_t, compute_eps_lower_single
from utils.clipbkd import craft_clipbkd, choose_worstcase_label
from utils.data import load_data


warnings.filterwarnings("ignore", message="PrivacyEngine detected new dataset object")

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def setup_device(local_rank: int):
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(device)
        return device
    return torch.device('cpu')


def xavier_init_model(model):
    def init_weights(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)

    model.apply(init_weights)


def init_wideresnet(model):
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


class AugmentationFunction:
    def __init__(self, image_size=32, channels=3):
        self.base_transforms = v2.Compose([
            v2.RandomCrop(image_size, padding=4),
            v2.RandomHorizontalFlip(p=0.5),
        ])

    def __call__(self, x):
        return self.base_transforms(x)


class AugmentedDataset(Dataset):
    def __init__(self, X, y, aug_fn, aug_mult=1, indices=None):
        self.X = X
        self.y = y
        self.aug_fn = aug_fn
        self.aug_mult = aug_mult
        self.indices = indices

    def __getitem__(self, index):
        x = self.X[index]
        y = self.y[index]
        idx = int(self.indices[index]) if self.indices is not None else index

        if self.aug_mult > 1 and self.aug_fn is not None:
            augmented = torch.stack([self.aug_fn(x) for _ in range(self.aug_mult)])
            return augmented, y, idx

        return x.unsqueeze(0), y, idx

    def __len__(self):
        return len(self.X)


def make_opacus_compatible(model):
    if not ModuleValidator.is_valid(model):
        model = ModuleValidator.fix(model)
    return model


def fgsm_attack(model, X, y, epsilon=0.1, max_iter=10, alpha=0.01):
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


def train_model_opacus(
    model_name,
    X,
    y,
    X_target,
    y_target,
    epsilon,
    delta,
    max_grad_norm,
    n_epochs,
    lr,
    batch_size,
    device,
    init_model=None,
    out_dim=10,
    aug_mult=1,
    defense=False,
    defense_k=5,
    world='in',
    rep=0,
    max_physical_batch_size=None,
    optimizer_name='sgd',
    early_stopping_patience=None,
):
    if init_model is None:
        model = Models[model_name](X.shape, out_dim=out_dim)
        if model_name == 'cnn':
            xavier_init_model(model)
        else:
            init_wideresnet(model)
    else:
        model = copy.deepcopy(init_model)

    model = make_opacus_compatible(model)
    model = model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss(reduction='none')

    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr)

    aug_fn = None
    if aug_mult > 1 and len(X.shape) > 2:
        aug_fn = AugmentationFunction(X.shape[2], X.shape[1])

    if aug_mult > 1 and aug_fn is not None:
        dataset = AugmentedDataset(X, y, aug_fn, aug_mult, indices=np.arange(len(X)))

        def collate_augmented(batch):
            xs = torch.stack([item[0] for item in batch])
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

    drop_mask = np.zeros(len(X), dtype=np.int8)
    scores = np.zeros(len(X), dtype=np.float32)

    use_private = epsilon is not None and max_grad_norm is not None
    privacy_engine = None

    canary_dropped_epoch = None
    canary_idx = len(X) - 1

    def create_loader(X_active, y_active, active_indices=None):
        if active_indices is None:
            active_indices = np.arange(len(X_active))

        if aug_mult > 1 and aug_fn is not None:
            ds = AugmentedDataset(X_active, y_active, aug_fn, aug_mult, indices=active_indices)

            def collate_augmented(batch):
                xs = torch.stack([item[0] for item in batch])
                ys = torch.tensor([item[1] for item in batch])
                idxs = torch.tensor([item[2] for item in batch])
                return xs, ys, idxs

            return DataLoader(
                ds,
                batch_size=min(batch_size, len(ds)),
                shuffle=True,
                drop_last=True,
                num_workers=0,
                collate_fn=collate_augmented,
            )

        if defense:
            ds = TensorDataset(X_active, y_active, torch.tensor(active_indices, dtype=torch.long))
        else:
            ds = TensorDataset(X_active, y_active)
        return DataLoader(
            ds,
            batch_size=min(batch_size, len(ds)),
            shuffle=True,
            drop_last=True,
            num_workers=0,
            pin_memory=True,
        )

    loader = create_loader(X, y)

    if use_private:
        effective_batch_size = min(batch_size, len(X))
        if effective_batch_size >= len(X):
            raise ValueError(
                "Invalid DP configuration: sample_rate must be < 1. "
                f"Got dataset_size={len(X)} and effective_batch_size={effective_batch_size} (batch_size={batch_size}). "
                "For tiny datasets (e.g. --n_df), set --batch_size <= dataset_size-1."
            )
        sample_rate = effective_batch_size / len(X)
        noise_multiplier = get_noise_multiplier(
            target_epsilon=epsilon,
            target_delta=delta,
            sample_rate=sample_rate,
            epochs=n_epochs,
            accountant='rdp',
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

    if max_physical_batch_size is None:
        max_physical_batch_size = batch_size

    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(n_epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        n_batches = 0

        active_mask = (drop_mask != 2)
        n_active = int(active_mask.sum())

        if defense and n_active == 0:
            break

        if aug_mult > 1 and aug_fn is not None:
            if use_private and max_physical_batch_size < batch_size:
                with BatchMemoryManager(
                    data_loader=loader,
                    max_physical_batch_size=max_physical_batch_size,
                    optimizer=optimizer,
                ) as memory_safe_loader:
                    for _, (curr_X, curr_y, idxs) in enumerate(memory_safe_loader):
                        B, A, C, H, W = curr_X.shape
                        curr_X_flat = curr_X.view(B * A, C, H, W).to(device, non_blocking=True)
                        curr_y_rep = curr_y.repeat_interleave(A).to(device, non_blocking=True)
                        idxs = idxs.to(device, non_blocking=True)

                        optimizer.zero_grad(set_to_none=True)
                        output = model(curr_X_flat)
                        loss_per_view = criterion(output, curr_y_rep)
                        loss = loss_per_view.view(B, A).mean(dim=1).mean()
                        loss.backward()

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
                            ascent_idxs = idxs.detach().cpu().numpy()[(drop_mask[idxs.detach().cpu().numpy()] == 1)]
                            drop_mask[ascent_idxs] = 2

                        epoch_loss += float(loss.item())
                        n_batches += 1
            else:
                for _, (curr_X, curr_y, idxs) in enumerate(loader):
                    B, A, C, H, W = curr_X.shape
                    curr_X_flat = curr_X.view(B * A, C, H, W).to(device, non_blocking=True)
                    curr_y_rep = curr_y.repeat_interleave(A).to(device, non_blocking=True)
                    idxs = idxs.to(device, non_blocking=True)

                    optimizer.zero_grad(set_to_none=True)
                    output = model(curr_X_flat)
                    loss_per_view = criterion(output, curr_y_rep)
                    loss = loss_per_view.view(B, A).mean(dim=1).mean()
                    loss.backward()

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
                        ascent_idxs = idxs.detach().cpu().numpy()[(drop_mask[idxs.detach().cpu().numpy()] == 1)]
                        drop_mask[ascent_idxs] = 2

                    epoch_loss += float(loss.item())
                    n_batches += 1
        else:
            if use_private and max_physical_batch_size < batch_size:
                with BatchMemoryManager(
                    data_loader=loader,
                    max_physical_batch_size=max_physical_batch_size,
                    optimizer=optimizer,
                ) as memory_safe_loader:
                    for _, batch in enumerate(memory_safe_loader):
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
                            ascent_idxs = idxs.detach().cpu().numpy()[(drop_mask[idxs.detach().cpu().numpy()] == 1)]
                            drop_mask[ascent_idxs] = 2

                        epoch_loss += float(loss.item())
                        n_batches += 1
            else:
                for _, batch in enumerate(loader):
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
                        ascent_idxs = idxs.detach().cpu().numpy()[(drop_mask[idxs.detach().cpu().numpy()] == 1)]
                        drop_mask[ascent_idxs] = 2

                    epoch_loss += float(loss.item())
                    n_batches += 1

        avg_loss = epoch_loss / n_batches if n_batches > 0 else 0.0

        epoch_time = time.time() - epoch_start
        if defense:
            print(
                f"[world={world} rep={rep}] Epoch: {epoch} (Active samples: {n_active}/{len(drop_mask)})"
                f" | Avg loss: {avg_loss:.6f} | Time: {epoch_time:.2f}s",
                flush=True,
            )
        else:
            print(
                f"[world={world} rep={rep}] Epoch: {epoch}"
                f" | Avg loss: {avg_loss:.6f} | Time: {epoch_time:.2f}s",
                flush=True,
            )

        if early_stopping_patience is not None:
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(
                        f"[world={world} rep={rep}] Early stopping at epoch {epoch} (patience={early_stopping_patience})",
                        flush=True,
                    )
                    break

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

            for idx in samples_to_mark:
                if drop_mask[idx] == 0:
                    drop_mask[idx] = 1

            active_indices = np.where(drop_mask != 2)[0]
            if len(active_indices) > 0:
                X_active = X[active_indices]
                y_active = y[active_indices]
                loader = create_loader(X_active, y_active, active_indices=active_indices)
                if use_private:
                    loader = privacy_engine._prepare_data_loader(loader, distributed=False, poisson_sampling=False)
            else:
                break

    return model, canary_dropped_epoch if defense else None


def test_model(model, X, y, device, batch_size=512):
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


def _gather_list_of_dicts(local_obj, world_size: int, rank: int):
    if not dist.is_available() or not dist.is_initialized() or world_size == 1:
        return [local_obj]
    gathered = [None for _ in range(world_size)] if rank == 0 else None
    dist.gather_object(local_obj, gathered, dst=0)
    return gathered


def distribute_reps(n_reps_half: int, world_size: int):
    reps_per_rank = [[] for _ in range(world_size)]
    for i in range(n_reps_half):
        reps_per_rank[i % world_size].append(i)
    return reps_per_rank


def main():
    parser = argparse.ArgumentParser(description='Audit DP-SGD using Opacus (one model per GPU/rank)')

    parser.add_argument('--data_name', type=str, default='mnist', help='dataset to use (mnist, cifar10, cifar100)')
    parser.add_argument('--model_name', type=str, default='cnn', choices=list(Models.keys()), help='model to audit')
    parser.add_argument('--n_reps', type=int, default=200, help='number of models to train')
    parser.add_argument('--n_df', type=int, default=0, help='|D| (0 => use full dataset)')
    parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--early_stopping', type=int, default=None,
                        help='early stopping patience (number of epochs without improvement)')
    parser.add_argument('--lr', type=float, default=1.33e-4, help='learning rate')
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam'], help='optimizer to use')
    parser.add_argument('--max_grad_norm', type=float, default=1, help='gradient clipping norm')
    parser.add_argument('--epsilon', type=float, default=10.0, help='privacy parameter, epsilon')
    parser.add_argument('--delta', type=float, default=1e-5, help='privacy parameter, delta')
    parser.add_argument('--target_type', type=str, default='blank', help='sample to use as target')
    parser.add_argument('--blank_alpha', type=float, default=0.0, help='interpolation factor for blank target')
    parser.add_argument('--seed', type=int, default=0, help='seed for reproducibility')
    parser.add_argument('--out', type=str, default='opacus_results_parallel/', help='folder to write results to')
    parser.add_argument('--fixed_init', type=str, nargs='?', default=None, const='',
                        help='initialize all models to the same weights')
    parser.add_argument('--batch_size', type=int, default=4000, help='batch size for training')
    parser.add_argument('--fit_world_only', type=str, default=None, choices=['in', 'out'],
                        help='just fit models in world and calculate losses')
    parser.add_argument('--alpha', type=float, default=0.05, help='significance level for empirical eps estimation')
    parser.add_argument('--badnets_label', type=int, default=-1, help='assign badnets poison this label')
    parser.add_argument('--view_badnets', action='store_true')
    parser.add_argument('--holdout_audit', action='store_true')
    parser.add_argument('--aug_mult', type=int, default=1, help='augmentation multiplier')
    parser.add_argument('--max_physical_batch_size', type=int, default=None,
                        help='max physical batch size for gradient accumulation')
    parser.add_argument('--linear_threshold', action='store_true',
                        help='use logistic regression to find threshold instead of exhaustive search')
    parser.add_argument('--defense', action='store_true', help='use filtering defense during audit')
    parser.add_argument('--defense_k', type=int, default=5, help='number of top samples to drop per class per epoch')

    args = parser.parse_args()

    world_size_env = os.environ.get('WORLD_SIZE', None)
    rank_env = os.environ.get('RANK', None)
    local_rank_env = os.environ.get('LOCAL_RANK', None)

    use_distributed = (
        world_size_env is not None
        and rank_env is not None
        and local_rank_env is not None
        and int(world_size_env) > 1
    )

    if use_distributed:
        dist.init_process_group(backend='nccl', init_method='env://')
        local_rank = int(local_rank_env)
        rank = int(rank_env)
        world_size = int(world_size_env)
    else:
        local_rank = 0
        rank = 0
        world_size = 1

    device = setup_device(local_rank)

    print(
        f"[Rank {rank}] world_size={world_size} local_rank={local_rank} device={device}",
        flush=True,
    )

    if args.max_grad_norm == -1:
        args.max_grad_norm = None

    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)

    out_folder = f'{args.out}/{args.data_name}_{args.model_name}_eps{args.epsilon}'
    if rank == 0:
        os.makedirs(out_folder, exist_ok=True)
        os.makedirs(f'{out_folder}/models', exist_ok=True)

    if dist.is_available() and dist.is_initialized() and world_size > 1:
        dist.barrier(device_ids=[local_rank] if torch.cuda.is_available() else None)

    if args.n_df == 1:
        X_out, y_out, out_dim = load_data(args.data_name, 1)
    else:
        X_out, y_out, out_dim = load_data(args.data_name, args.n_df - 1)

    init_model = None
    if args.fixed_init is not None:
        init_model = Models[args.model_name](X_out.shape, out_dim=out_dim)
        if args.fixed_init == '':
            if args.model_name == 'cnn':
                xavier_init_model(init_model)
            else:
                init_wideresnet(init_model)
        else:
            init_model.load_state_dict(torch.load(args.fixed_init, map_location='cpu'))
            X_out, y_out = X_out[len(X_out) // 2:], y_out[len(y_out) // 2:]

    if args.target_type == 'blank':
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
        target_X = X_out[-1].clone()
        target_y = torch.tensor(args.badnets_label)
        target_X[:, -4:, -4:] = torch.max(target_X)
        target_X = target_X.unsqueeze(0)
        target_y = target_y.unsqueeze(0)
        if args.view_badnets and rank == 0:
            plt.imshow(target_X.squeeze().numpy(), cmap='gray')
            plt.savefig(f'badnets_{args.badnets_label}.png')
    elif args.target_type == 'sanity_check':
        target_X = X_out[-1].unsqueeze(0)
        target_y = y_out[-1].unsqueeze(0)
    elif args.target_type == 'clipbkd':
        target_X, target_y = craft_clipbkd(X_out, init_model)
    elif args.target_type == 'fgsm':
        fgsm_model = Models[args.model_name](X_out.shape, out_dim=out_dim).to(device)
        if args.model_name == 'cnn':
            xavier_init_model(fgsm_model)
        else:
            init_wideresnet(fgsm_model)

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
            device=device,
            init_model=fgsm_model,
            out_dim=out_dim,
            aug_mult=1,
            defense=False,
            defense_k=args.defense_k,
            world='out',
            rep=0,
            max_physical_batch_size=args.max_physical_batch_size,
            optimizer_name=args.optimizer,
            early_stopping_patience=args.early_stopping,
        )

        original_X = X_out[-1].unsqueeze(0).to(device)
        original_y = y_out[-1].unsqueeze(0).to(device)
        target_class = (original_y + 1) % out_dim

        target_X, _ = fgsm_attack(fgsm_model, original_X, target_class, epsilon=0.1, max_iter=20, alpha=0.01)
        target_y = target_class
        target_X = target_X.cpu()
        target_y = target_y.cpu()
    elif os.path.exists(args.target_type):
        if args.target_type.endswith('.pt'):
            canary_data = torch.load(args.target_type, map_location='cpu')
            target_X = canary_data['canary'].unsqueeze(0)
            target_y = torch.tensor([canary_data['audit_label']])
            if 'init_model' in canary_data and args.fixed_init is not None and init_model is not None:
                init_model.load_state_dict(canary_data['init_model'])
        else:
            target_X = torch.from_numpy(np.load(args.target_type))
            if init_model is not None:
                target_y = choose_worstcase_label(init_model, target_X)
            else:
                target_y = torch.from_numpy(np.array([9]))
            if target_X.ndim == X_out.ndim - 1:
                target_X = target_X.unsqueeze(0)
    else:
        raise Exception(f'Target {args.target_type} not found')

    X_in = torch.vstack((X_out[:-1], target_X))
    y_in = torch.cat((y_out[:-1], target_y))

    X_test, y_test, _ = load_data(args.data_name, None, split='test')

    y_out = y_out.long()
    y_in = y_in.long()
    y_test = y_test.long()

    worlds = [args.fit_world_only] if args.fit_world_only else ['in', 'out']

    n_reps_half = args.n_reps // 2
    reps_per_rank = distribute_reps(n_reps_half, world_size)
    my_reps = reps_per_rank[rank]

    local_results = {'in': [], 'out': []}
    local_drop_epochs = []
    local_accs = []

    for world in worlds:
        curr_X, curr_y = (X_out, y_out) if world == 'out' else (X_in, y_in)
        for rep in my_reps:
            rep_start = time.time()
            print(f"[Rank {rank}] START world={world} rep={rep}", flush=True)
            rep_seed = args.seed + rank * 100000 + rep
            np.random.seed(rep_seed)
            torch.manual_seed(rep_seed)
            torch.cuda.manual_seed_all(rep_seed)

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
                device=device,
                init_model=init_model,
                out_dim=out_dim,
                aug_mult=args.aug_mult,
                defense=args.defense,
                defense_k=args.defense_k,
                world=world,
                rep=rep,
                max_physical_batch_size=args.max_physical_batch_size,
                optimizer_name=args.optimizer,
                early_stopping_patience=args.early_stopping,
            )

            model.eval()
            with torch.no_grad():
                dev = next(model.parameters()).device
                output = model(target_X.to(dev))
                score = -nn.CrossEntropyLoss()(output, target_y.to(dev)).cpu().item()

            local_results[world].append((rep, output[0].detach().cpu().numpy(), float(score)))
            if args.defense and world == 'in':
                local_drop_epochs.append((rep, int(canary_dropped_epoch) if canary_dropped_epoch is not None else -1))

            # Match opacus_audit.py behavior: compute train/test acc for the first 5 reps in the IN world.
            if world == 'in' and rep < 5:
                train_acc = test_model(model, X_in, y_in, device=device)
                test_acc = test_model(model, X_test, y_test, device=device)
                local_accs.append((rep, float(train_acc), float(test_acc)))

            print(
                f"[Rank {rank}] END world={world} rep={rep} elapsed_s={time.time() - rep_start:.2f}",
                flush=True,
            )

    gathered = _gather_list_of_dicts(local_results, world_size, rank)
    gathered_drop = _gather_list_of_dicts(local_drop_epochs, world_size, rank)
    gathered_accs = _gather_list_of_dicts(local_accs, world_size, rank)

    if rank == 0:
        outputs = {'in': [], 'out': []}
        losses = {'in': [], 'out': []}

        for world in worlds:
            merged = []
            for r in range(world_size):
                merged.extend(gathered[r][world])
            merged.sort(key=lambda t: t[0])

            outputs[world] = np.array([x[1] for x in merged])
            losses[world] = np.array([x[2] for x in merged], dtype=np.float32)

        train_set_accs = []
        test_set_accs = []
        if 'in' in worlds:
            merged_accs = []
            for r in range(world_size):
                merged_accs.extend(gathered_accs[r])
            merged_accs.sort(key=lambda t: t[0])

            # Fill arrays for reps 0..4 if present
            rep_to_acc = {int(rep): (float(tr), float(te)) for rep, tr, te in merged_accs}
            for rep in range(5):
                if rep in rep_to_acc:
                    tr, te = rep_to_acc[rep]
                    train_set_accs.append(tr)
                    test_set_accs.append(te)

        np.save(f'{out_folder}/train_set_accs.npy', np.asarray(train_set_accs, dtype=np.float32))
        np.save(f'{out_folder}/test_set_accs.npy', np.asarray(test_set_accs, dtype=np.float32))

        np.save(f'{out_folder}/outputs_in.npy', outputs.get('in', np.zeros((0, out_dim), dtype=np.float32)))
        np.save(f'{out_folder}/outputs_out.npy', outputs.get('out', np.zeros((0, out_dim), dtype=np.float32)))
        np.save(f'{out_folder}/losses_in.npy', losses.get('in', np.zeros((0,), dtype=np.float32)))
        np.save(f'{out_folder}/losses_out.npy', losses.get('out', np.zeros((0,), dtype=np.float32)))

        if args.defense:
            merged_drop = []
            for r in range(world_size):
                merged_drop.extend(gathered_drop[r])
            merged_drop.sort(key=lambda t: t[0])
            canary_drop_epochs = np.array([x[1] for x in merged_drop], dtype=np.int32)
            np.save(f'{out_folder}/canary_drop_epochs.npy', canary_drop_epochs)

        if not args.fit_world_only:
            def compute_eps_with_linear_threshold(scores, labels, alpha, delta):
                scores = np.array(scores).reshape(-1, 1)
                labels = np.array(labels)

                clf = LogisticRegression(solver='lbfgs', max_iter=1000)
                clf.fit(scores, labels)

                w = clf.coef_[0][0]
                b = clf.intercept_[0]
                if abs(w) > 1e-10:
                    threshold = -b / w
                else:
                    threshold = np.median(scores)

                predictions = (scores.flatten() >= threshold).astype(int)
                tp = np.sum((predictions == 1) & (labels == 1))
                fp = np.sum((predictions == 1) & (labels == 0))
                tn = np.sum((predictions == 0) & (labels == 0))
                fn = np.sum((predictions == 0) & (labels == 1))

                results = AttackResults(FN=fn, FP=fp, TN=tn, TP=tp)
                emp_eps = compute_eps_lower_single(results, alpha, delta, method='GDP')
                return threshold, emp_eps

            k = len(losses['in'])
            if args.holdout_audit:
                k = k // 2

            t_in = losses['in'][:k]
            t_out = losses['out'][:k]
            h_in = losses['in'][k:]
            h_out = losses['out'][k:]

            mia_scores = np.concatenate([t_in, t_out])
            mia_labels = np.concatenate([np.ones_like(t_in), np.zeros_like(t_out)])

            if args.linear_threshold:
                max_t, emp_eps_loss = compute_eps_with_linear_threshold(mia_scores, mia_labels, args.alpha, args.delta)
            else:
                max_t, emp_eps_loss = compute_eps_lower_from_mia(mia_scores, mia_labels, args.alpha, args.delta, 'GDP', n_procs=1)

            if args.holdout_audit:
                emp_eps_loss = compute_eps_lower_from_mia_given_t(
                    np.concatenate([h_in, h_out]),
                    np.concatenate([np.ones_like(h_in), np.zeros_like(h_out)]),
                    args.alpha,
                    args.delta,
                    max_t,
                    'GDP',
                )

            np.save(f'{out_folder}/emp_eps_loss.npy', [emp_eps_loss])
            np.save(f'{out_folder}/mia_scores.npy', mia_scores)
            np.save(f'{out_folder}/mia_labels.npy', mia_labels)

            print(f'Theoretical eps: {args.epsilon}')
            print(f'Empirical eps: {emp_eps_loss}')

    if dist.is_available() and dist.is_initialized() and world_size > 1:
        dist.barrier(device_ids=[local_rank] if torch.cuda.is_available() else None)
        dist.destroy_process_group()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nInterrupted by user')
    except Exception as e:
        print(f'\nError: {str(e)}')
        import traceback
        traceback.print_exc()
