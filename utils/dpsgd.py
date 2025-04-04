"""
Utility functions to execute DP-SGD
"""
import torch
import numpy as np
from torch.func import functional_call, vmap, grad

# NOTE: potentially get rid of this
from defense_utils import *

def get_per_sample_grads(model, X, y, criterion):
    """Compute per-sample gradients"""

    # map of parameter names : parameter values
    params = {k: v.detach() for k, v in model.named_parameters()}
    # map of buffer names : buffer balues
    buffers = {k: v.detach() for k, v in model.named_buffers()}
    
    def compute_loss(params, buffers, sample, target):
        batch = sample.unsqueeze(0)
        targets = target.unsqueeze(0)

        predictions = functional_call(model, (params, buffers), (batch,))
        loss = criterion(predictions, targets)
        return loss
    
    ft_compute_grad = grad(compute_loss)
    ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0, 0))
    ps_grads = ft_compute_sample_grad(params, buffers, X, y)

    return ps_grads


def _get_per_sample_grads(model, X, y, criterion):
    model.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    ps_grads = {name: param.grad_sample for name, param in model.named_parameters()}
    embeddings = model._module.embeddings
    return ps_grads, embeddings
    

def get_per_sample_grad_norms(per_sample_grads):
    """Compute L2 norms of per-sample gradients"""
    return torch.vstack([
        curr_grad.flatten(start_dim=1).norm(2, dim=1)
        for curr_grad in per_sample_grads.values()
    ]).norm(2, dim=0)


def clip_per_sample_grads(per_sample_grads, max_grad_norm):
    """Clip per-sample gradients to clipping norm"""
    ps_grad_norms = get_per_sample_grad_norms(per_sample_grads)

    ps_grad_scales = 1 / torch.maximum(
        torch.ones_like(ps_grad_norms),
        ps_grad_norms / max_grad_norm
    )

    ps_grads_clipped = {
        # broadcast
        name: curr_grad * ps_grad_scales[(...,) + (None,) * (curr_grad.dim() - 1)]
        for name, curr_grad in per_sample_grads.items() 
    }

    ps_grad_norms_clipped = get_per_sample_grad_norms(ps_grads_clipped)

    return ps_grads_clipped, { 'before': ps_grad_norms, 'after': ps_grad_norms_clipped.cpu().numpy() }


def clip_and_accum_grads_block(model, X, y, optimizer, criterion, max_grad_norm):
    """Clip and accumulate gradients of a single block of samples"""
    optimizer.zero_grad()

    if len(X) == 0:
        # empty dataset
        ps_grads = { name: torch.zeros_like(param).unsqueeze(dim=0) for name, param in model.named_parameters() }
    else:
        # calculate per-sample gradients
        # ps_grads = get_per_sample_grads(model, X, y, criterion)
        # embeddings = torch.tensor(np.array([])).to(X.device)

        ps_grads, embeddings = _get_per_sample_grads(model, X, y, criterion)

    ps_grad_norms_data = { 'before': np.array([]), 'after': np.array([]) }
    if max_grad_norm is not None:
        # clip per-sample gradients
        ps_grads_clipped, ps_grad_norms_data = clip_per_sample_grads(ps_grads, max_grad_norm)
    else:
        ps_grads_clipped = ps_grads

    with torch.no_grad():
        accum_grad_block = {name: grad.sum(dim=0) for name, grad in ps_grads_clipped.items()}

    return embeddings, accum_grad_block, ps_grads, ps_grad_norms_data


def global_clip_and_accum_grads(model, X, y, optimizer, criterion, max_grad_norm, block_size=1024, use_defense=False, spectral_signature_args=None):
    """Clip and return per-sample gradients"""

    find_outliers = False
    drop_mask = None

    if spectral_signature_args:
        drop_mask = spectral_signature_args['drop_mask']
        find_outliers = True

    # split samples into blocks by index
    idx_blocks = torch.split(torch.from_numpy(np.arange(len(X))), block_size)

    accum_grad = None

    all_embeddings = []
    all_ps_grads = {name: [] for name, _ in model.named_parameters()}
    all_ps_grad_norms_data_before = []

    for i, idx_block in enumerate(idx_blocks):
        # get a single block of samples
        block_drop_mask = drop_mask[idx_block]
        idx_block = idx_block[block_drop_mask == 0].to(y.device)
        curr_X, curr_y = X[idx_block], y[idx_block]

        optimizer.zero_grad()

        ps_grads, embeddings = _get_per_sample_grads(model, curr_X, curr_y, criterion)
        ps_grads_clipped, ps_grad_norms_data = clip_per_sample_grads(ps_grads, max_grad_norm)

        all_embeddings.append(embeddings.cpu().numpy())
        for name, _ in model.named_parameters():
            all_ps_grads[name].append(ps_grads[name].cpu().numpy())
        all_ps_grad_norms_data_before.append(ps_grad_norms_data['before'].cpu().numpy())

        with torch.no_grad():
            accum_grad_block = {name: grad.sum(dim=0) for name, grad in ps_grads_clipped.items()}

        # accum grads for all blocks
        if accum_grad is None:
            accum_grad = accum_grad_block
        else:
            with torch.no_grad():
                for name, curr_grad in accum_grad_block.items():
                    accum_grad[name] = accum_grad[name] + curr_grad

    all_embeddings = np.concatenate(all_embeddings)
    for name, grads in all_ps_grads.items():
        all_ps_grads[name] = np.concatenate(grads)
    all_ps_grad_norms_data_before = np.concatenate(all_ps_grad_norms_data_before)

    if find_outliers:
        spectral_space = None
        if spectral_signature_args['search_space'] == 'embedding':
            spectral_space = all_embeddings
        elif spectral_signature_args['search_space'] == 'gradient':

            if spectral_signature_args['scoring_fn'] in ['norm', 'whitened_norm', 'pca', 'walign']:
                last_layer_name = list(model._module.net.named_modules())[-1][0]
                last_w_name = '_module.net.' + last_layer_name + '.weight'
                last_b_name = '_module.net.' + last_layer_name + '.bias'
                w = [t.flatten() for t in all_ps_grads[last_w_name]]
                w = np.stack(w)
                b = [t.flatten() for t in all_ps_grads[last_b_name]]
                b = np.stack(b)
                last_layer_grads = np.concatenate((w, b), axis=1)
                spectral_space = last_layer_grads

                accum_spectral_space = torch.cat([accum_grad_block[last_w_name].flatten(), accum_grad_block[last_b_name].flatten()], dim=0).unsqueeze(0).cpu().numpy()


        cpu_y = y.cpu().numpy()[drop_mask == 0]
        all_scores = np.zeros_like(cpu_y)

        if spectral_signature_args['scoring_fn'] == 'full_model_norm':
            all_scores = np.flip(np.argsort(all_ps_grad_norms_data_before))
        else:
            for k in range(0, 10):
                k_spectral_space = spectral_space[cpu_y == k]
                if len(k_spectral_space) == 0: continue

                k_spectral_space = k_spectral_space - k_spectral_space.mean(axis=0, keepdims=True)

                _, principals_components = PCA_np(k_spectral_space)
                pc1 = np.expand_dims(principals_components[:, -1], axis=0)

                if spectral_signature_args['scoring_fn'] == 'whitened_norm':
                    W = whiten_np(k_spectral_space)
                    k_spectral_space = k_spectral_space @ W

                if spectral_signature_args['scoring_fn'] == 'pca':
                    scores = (k_spectral_space @ pc1.T).flatten() ** 2
                elif spectral_signature_args['scoring_fn'] in ['norm', 'whitened_norm']:
                    scores = np.linalg.norm(k_spectral_space, axis=1)
                elif spectral_signature_args['scoring_fn'] == 'walign':
                    accum_spectral_space /= np.linalg.norm(accum_spectral_space)
                    k_spectral_space /= np.linalg.norm(k_spectral_space, axis=1, keepdims=True)
                    scores = (k_spectral_space @ accum_spectral_space.T).flatten() ** 2
                
                all_scores[cpu_y == k] = scores
            
            all_scores_idx = np.flip(np.argsort(all_scores))

            print(all_scores)
            print('Canary Rank:', np.where(all_scores_idx == len(all_scores_idx) - 1))

            if use_defense:
                for idx in all_scores_idx:
                    drop_mask[drop_mask == 0][idx] = 1
                    for name, _ in model.named_parameters():
                        clipping_factor = 1 / max(1, all_ps_grad_norms_data_before[idx] / max_grad_norm)
                        accum_grad[name] -= clipping_factor * all_ps_grads[name][idx]

    return accum_grad



def local_clip_and_accum_grads(model, X, y, optimizer, criterion, max_grad_norm, block_size=1024, use_defense=False, spectral_signature_args=None):
    """Clip and accumulate gradients in blocks of samples to conserve gpu space"""

    find_outliers = False
    drop_mask = torch.zeros_like(y).to(y.device)

    if spectral_signature_args:
        drop_mask = spectral_signature_args['drop_mask']
        find_outliers = True

    # split samples into blocks by index
    idx_blocks = torch.split(torch.from_numpy(np.arange(len(X))), block_size)

    accum_grad = None

    outlier_grads = {name:[] for name, _ in model.named_parameters()}
    outlier_scores = []
    outlier_indices = []

    for i, idx_block in enumerate(idx_blocks):
        # get a single block of samples
        idx_block = idx_block.to(y.device)
        block_drop_mask = drop_mask[idx_block]

        curr_X, curr_y = X[idx_block[block_drop_mask == 0]], y[idx_block[block_drop_mask == 0]]

        optimizer.zero_grad()

        ps_grads, curr_embeddings = _get_per_sample_grads(model, curr_X, curr_y, criterion)
        ps_grads_clipped, ps_grad_norms_data = clip_per_sample_grads(ps_grads, max_grad_norm)

        with torch.no_grad():
            accum_grad_block = {name: grad.sum(dim=0) for name, grad in ps_grads_clipped.items()}
        
        # accum grads for all blocks
        if accum_grad is None:
            accum_grad = accum_grad_block
        else:
            with torch.no_grad():
                for name, curr_grad in accum_grad_block.items():
                    accum_grad[name] = accum_grad[name] + curr_grad
        ###################################################

        if find_outliers:
            with torch.no_grad():

                spectral_space = None

                if spectral_signature_args['search_space'] == 'embedding':
                    spectral_space = curr_embeddings
                elif spectral_signature_args['search_space'] == 'gradient':
                    if spectral_signature_args['scoring_fn'] in ['norm', 'whitened_norm', 'pca', 'walign']:
                        last_layer_name = list(model._module.net.named_modules())[-1][0]
                        last_w_name = '_module.net.' + last_layer_name + '.weight'
                        last_b_name = '_module.net.' + last_layer_name + '.bias'

                        w = [t.flatten() for t in ps_grads[last_w_name]]
                        w = torch.stack(w)
                        b = [t.flatten() for t in ps_grads[last_b_name]]
                        b = torch.stack(b)
                        last_layer_grads = torch.cat((w, b), dim=1)
                        spectral_space = last_layer_grads

                        accum_spectral_space = torch.cat([accum_grad_block[last_w_name].flatten(), accum_grad_block[last_b_name].flatten()], dim=0).unsqueeze(0)
                        accum_spectral_space = accum_spectral_space - torch.mean(accum_spectral_space)

                all_scores_in_block = torch.zeros_like(curr_y, dtype=torch.float32).to(y.device)

                if spectral_signature_args['scoring_fn'] == 'full_model_norm':
                    all_scores_in_block = ps_grad_norms_data['before']
                else:
                    for k in range(0, 10):
                        k_spectral_space = spectral_space[curr_y == k]
                        if len(k_spectral_space) == 0: continue
                        
                        # Center last layer gradients
                        k_spectral_space = k_spectral_space - k_spectral_space.mean(dim=0, keepdim=True)

                        # Whiten last layer gradients
                        if spectral_signature_args['scoring_fn'] in ['whitened_norm', 'clipped_whitened_norm']:
                            W = whiten(k_spectral_space)
                            k_spectral_space = k_spectral_space @ W

                        if spectral_signature_args['scoring_fn'] == 'pca':
                            _, principal_components = PCA(k_spectral_space)
                            pc1 = principal_components[:, -1].unsqueeze(0)
                            scores = (k_spectral_space @ pc1.T).flatten() ** 2
                        elif spectral_signature_args['scoring_fn'] in ['norm', 'whitened_norm']:
                            scores = torch.linalg.norm(k_spectral_space, dim=1)
                        elif spectral_signature_args['scoring_fn'] == 'walign':
                            # accum_spectral_space /= torch.linalg.norm(accum_spectral_space)
                            # k_spectral_space /= torch.linalg.norm(k_spectral_space, dim=1, keepdim=True)
                            scores = (k_spectral_space @ accum_spectral_space.T).flatten() ** 2

                        all_scores_in_block[curr_y == k] = scores

                all_scores_in_block_idx = torch.argsort(all_scores_in_block, descending=True)

                if len(y) - 1 in idx_block:
                    loc = torch.where(all_scores_in_block_idx == (len(all_scores_in_block_idx) - 1))[0]
                    print(loc, all_scores_in_block[-1], all_scores_in_block[all_scores_in_block_idx[0]], all_scores_in_block[all_scores_in_block_idx[1]], all_scores_in_block[all_scores_in_block_idx[2]])

                for idx in all_scores_in_block_idx[:30]:
                    outlier_scores.append(all_scores_in_block[idx])
                    outlier_indices.append(idx_block[idx])
                    for name, _ in model.named_parameters():
                        outlier_grads[name].append((ps_grads_clipped[name][idx]).cpu().numpy())

        ###################################################


    if find_outliers:
        outlier_scores = torch.tensor(outlier_scores).to(y.device)
        outlier_scores_idx = torch.argsort(outlier_scores, descending=True)
        outlier_indices = torch.tensor(outlier_indices).to(y.device)[outlier_scores_idx]
        print('Canary Rank in Epoch-wide Outliers:', torch.where(outlier_indices == len(X) - 1))
        if spectral_signature_args['store_canary_rank'] is not None:
            rank = torch.where(outlier_indices == len(X) - 1)
            if len(rank) == 0 or len(rank[0]) == 0: rank = -1
            else:
                rank = rank[0].item()
            spectral_signature_args['store_canary_rank'].append(rank)
        if use_defense:
            drop_mask[drop_mask == 0][outlier_indices] = 1

            # Option 1: Literally subtract out the outlier gradient
            for idx in outlier_scores_idx[:30]:
                for name, _ in model.named_parameters():
                    accum_grad[name] -= torch.tensor(outlier_grads[name][idx]).to(y.device)

            # Option 2: Use inverse of outlier scores as weights when aggregating gradients

    return accum_grad