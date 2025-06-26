"""
Utility functions to execute DP-SGD
"""
import torch
import numpy as np
from torch.func import functional_call, vmap, grad
import matplotlib.pyplot as plt

from opacus.accountants.utils import get_noise_multiplier

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

    N_drop = int(0.05 * len(y) // 100)

    find_outliers = False
    canary_dropped = False
    out = ''
    drop_mask = np.zeros(len(y))

    if spectral_signature_args:
        drop_mask = spectral_signature_args['drop_mask']
        out = spectral_signature_args['out']
        find_outliers = True

    # split samples into blocks by index
    idx_blocks = torch.split(torch.from_numpy(np.arange(len(X))), block_size)

    accum_grad = None

    last_w_name, last_b_name = '', ''
    if spectral_signature_args['search_space'] == 'gradient':
        last_layer_name = list(model.net.named_modules())[-1][0]
        last_w_name = 'net.' + last_layer_name + '.weight'
        last_b_name = 'net.' + last_layer_name + '.bias'

    all_embeddings = []
    all_ps_grads = {name: [] for name, _ in model.named_parameters()}
    all_ps_grad_norms_data_before = []

    for i, idx_block in enumerate(idx_blocks):
        # get a single block of samples
        block_drop_mask = drop_mask[idx_block]
        idx_block = idx_block[block_drop_mask == 0].to(y.device)
        curr_X, curr_y = X[idx_block], y[idx_block]

        optimizer.zero_grad()

        if spectral_signature_args['search_space'] == 'embedding':
            ps_grads, curr_embeddings = _get_per_sample_grads(model, curr_X, curr_y, criterion)
        else:
            ps_grads = get_per_sample_grads(model, curr_X, curr_y, criterion)
        ps_grads_clipped, ps_grad_norms_data = clip_per_sample_grads(ps_grads, max_grad_norm)

        with torch.no_grad():
            accum_grad_block = {name: grad.sum(dim=0) for name, grad in ps_grads_clipped.items()}

        # if find_outliers:
        #     del ps_grads_clipped
        #     torch.cuda.empty_cache()

        # accum grads for all blocks
        if accum_grad is None:
            accum_grad = accum_grad_block
        else:
            with torch.no_grad():
                for name, curr_grad in accum_grad_block.items():
                    accum_grad[name] = accum_grad[name] + curr_grad


        if spectral_signature_args['search_space'] == 'embedding':
            all_embeddings.append(curr_embeddings.cpu().numpy())
        else:
            for name, _ in model.named_parameters():
                all_ps_grads[name].append(ps_grads[name].cpu().numpy())

            all_ps_grad_norms_data_before.append(ps_grad_norms_data['before'].cpu().numpy())

    if spectral_signature_args['search_space'] == 'embedding':
        all_embeddings = np.concatenate(all_embeddings)

    for name, grads in all_ps_grads.items():
        all_ps_grads[name] = np.concatenate(grads)

    all_ps_grad_norms_data_before = np.concatenate(all_ps_grad_norms_data_before)
    
    if find_outliers:
        cpu_y = y.cpu().numpy()[drop_mask == 0]
        all_scores = np.zeros_like(cpu_y, dtype=float)

        with torch.no_grad():
            for k in range(0, 10):
                if spectral_signature_args['search_space'] == 'embeddings':
                    k_spectral_space = all_embeddings[cpu_y == k]
                else:
                    mask = cpu_y == k
                    grads_w = all_ps_grads[last_w_name][mask]
                    grads_b = all_ps_grads[last_b_name][mask]
                    w = np.vstack([g[k] for g in grads_w])
                    b = np.vstack([g[k] for g in grads_b])

                    k_spectral_space = np.concatenate((w, b), axis=1)
                
                # Center spectral space
                k_spectral_space = k_spectral_space - k_spectral_space.mean(axis=0, keepdims=True)
                
                if spectral_signature_args['scoring_fn'] == 'whitened_norm':
                    W = whiten_np(k_spectral_space)
                    k_spectral_space = k_spectral_space @ W
                    scores = np.linalg.norm(k_spectral_space, axis=1)

                elif spectral_signature_args['scoring_fn'] == 'norm':
                    scores = np.linalg.norm(k_spectral_space, axis=1)

                elif spectral_signature_args['scoring_fn'] == 'scaled_norm':
                    k_std = np.std(k_spectral_space, axis=0, keepdims=True, ddof=0)
                    k_spectral_space = k_spectral_space / k_std
                    scores = torch.linalg.norm(k_spectral_space, axis=1)

                elif spectral_signature_args['scoring_fn'] == 'pca':
                    _, _, vt = np.linalg.svd(k_spectral_space, full_matrices=False)
                    pc1 = vt[0].reshape(1, -1)
                    projections = pc1 @ k_spectral_space.T
                    scores = np.linalg.norm(projections, axis=0)

                elif spectral_signature_args['scoring_fn'] == 'scaled_pca':
                    k_std = np.std(k_spectral_space, axis=0, keepdims=True, ddof=0)
                    k_spectral_space = k_spectral_space / k_std
                    _, _, vt = np.linalg.svd(k_spectral_space, full_matrices=False)
                    pc1 = vt[0].reshape(1, -1)
                    projections = pc1 @ k_spectral_space.T
                    scores = np.linalg.norm(projections, axis=0)

                else:
                    raise NotImplementedError

                all_scores[cpu_y == k] = scores
            
            all_scores_idx = np.flip(np.argsort(all_scores))

            canary_rank = np.where(all_scores_idx == (len(all_scores_idx) - 1))[0]

            if spectral_signature_args['store_canary_rank'] is not None:
                print('Canary Rank:', canary_rank)
                if not canary_dropped and canary_rank < N_drop:
                    _, _, vt = np.linalg.svd(k_spectral_space, full_matrices=False)
                    projections = k_spectral_space @ vt[:2].T
                    plt.clf()
                    plt.figure()
                    plt.scatter(projections[:, 0], projections[:, 1], alpha=0.6, color='blue')
                    plt.scatter(projections[-1, 0], projections[-1, 1], color='red', label='Canary')               
                    plt.xlabel('PC1')
                    plt.ylabel('PC2')
                    plt.title('Projection onto First 2 Components of Spectral Space')
                    plt.legend()
                    plt.grid(True)
                    plt.savefig(f'{out}/spectral_sig_viz.png')
                    # Plot the score distribution on the iteration where the canary is dropped
                    plt.clf()
                    plt.figure()
                    plt.plot(sorted(all_scores), color='blue')
                    plt.plot(len(all_scores) - N_drop - 1, sorted(all_scores)[-1 * N_drop - 1].item(), marker='o', color='green', markersize=10, label='Smallest Dropped Score')
                    plt.plot(len(all_scores) - 1, all_scores[-1].item(), marker='o', color='red', markersize=10, label='Canary Score')

                    plt.title("Sorted Scores (Outliers will pop)")
                    plt.yscale('log')
                    plt.ylabel("Score")
                    plt.legend()
                    plt.grid(True)
                    plt.savefig(f'{out}/score_dist.png')

                    canary_dropped = True

                spectral_signature_args['store_canary_rank'].append(canary_rank)

            if use_defense != '':
                abs_score_idx = np.flatnonzero(~drop_mask)[all_scores_idx[:N_drop]]
                drop_mask[abs_score_idx] = 1
                for idx in all_scores_idx[:N_drop]:
                    for name, _ in model.named_parameters():
                        clipping_factor = 1 / max(1, all_ps_grad_norms_data_before[idx] / max_grad_norm)

                        if use_defense['drop']:
                            accum_grad[name] -= clipping_factor * torch.tensor(ps_grads[name][idx], dtype=torch.float32).to(y.device)
                        elif use_defense == 'grad_ascent':
                            accum_grad[name] -= 2 * clipping_factor * torch.tensor(ps_grads[name][idx], dtype=torch.float32).to(y.device)

    return accum_grad


# TODO: switch back from ps_grads_clipped, get rid of noised gradients
def local_clip_and_accum_grads(model, X, y, optimizer, criterion, max_grad_norm, block_size=1024, use_defense=False, spectral_signature_args=None):
    """Clip and accumulate gradients in blocks of samples to conserve gpu space"""
    
    n_epochs = 100 if spectral_signature_args is None else spectral_signature_args['n_epochs']
    N_drop = int(0.05 * len(y) // (n_epochs * (len(y) // block_size)))

    find_outliers = False
    canary_dropped = False
    out = ''
    drop_mask = torch.zeros_like(y)

    if spectral_signature_args:
        drop_mask = spectral_signature_args['drop_mask']
        out = spectral_signature_args['out']
        canary_dropped = spectral_signature_args['canary_dropped']
        find_outliers = True

    # split samples into blocks by index
    idx_blocks = torch.split(torch.from_numpy(np.arange(len(X))), block_size)

    accum_grad = None

    last_w_name, last_b_name = '', ''
    if spectral_signature_args and spectral_signature_args['search_space'] == 'gradient':
        last_layer_name = list(model.net.named_modules())[-1][0]
        last_w_name = 'net.' + last_layer_name + '.weight'
        last_b_name = 'net.' + last_layer_name + '.bias'

    for i, idx_block in enumerate(idx_blocks):
        # get a single block of samples
        idx_block = idx_block.to(y.device)
        block_drop_mask = drop_mask[idx_block]

        curr_X, curr_y = X[idx_block[block_drop_mask == 0]], y[idx_block[block_drop_mask == 0]]

        optimizer.zero_grad()

        ps_grads = None
        if spectral_signature_args and spectral_signature_args['search_space'] == 'embedding':
            ps_grads, curr_embeddings = _get_per_sample_grads(model, curr_X, curr_y, criterion)
        else:
            ps_grads = get_per_sample_grads(model, curr_X, curr_y, criterion)
        ps_grads_clipped, ps_grad_norms_data = None, None
        if max_grad_norm is not None:
            ps_grads_clipped, ps_grad_norms_data = clip_per_sample_grads(ps_grads, max_grad_norm)
        else:
            ps_grads_clipped = ps_grads

        with torch.no_grad():
            accum_grad_block = {name: grad.sum(dim=0) for name, grad in ps_grads_clipped.items()}
        
        # if find_outliers:
        #     del ps_grads_clipped
        #     torch.cuda.empty_cache()

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
                all_scores_in_block = torch.zeros_like(curr_y, dtype=torch.float32)

                for k in range(0, 10):
                    if spectral_signature_args['search_space'] == 'embedding':
                        k_spectral_space = curr_embeddings[curr_y == k]
                    else:
                        mask = curr_y == k
                        grads_w = ps_grads[last_w_name][mask]
                        grads_b = ps_grads[last_b_name][mask]
                        w = torch.vstack([g[k] for g in grads_w])
                        b = torch.vstack([g[k] for g in grads_b])

                        k_spectral_space = torch.cat((w, b), dim=1)

                    if len(k_spectral_space) == 0: continue
                    
                    # Center spectral space
                    k_spectral_space = k_spectral_space - k_spectral_space.mean(dim=0, keepdim=True)
                    
                    if spectral_signature_args['scoring_fn'] == 'whitened_norm':
                        W = whiten(k_spectral_space)
                        k_spectral_space = k_spectral_space @ W
                        scores = torch.linalg.norm(k_spectral_space, dim=1)

                    elif spectral_signature_args['scoring_fn'] == 'norm':
                        scores = torch.linalg.norm(k_spectral_space, dim=1, ord=1)

                    elif spectral_signature_args['scoring_fn'] == 'scaled_norm':
                        k_std = k_spectral_space.std(dim=0, keepdim=True, unbiased=False)
                        k_spectral_space = k_spectral_space / k_std
                        scores = torch.linalg.norm(k_spectral_space, dim=1)

                    elif spectral_signature_args['scoring_fn'] == 'pca':
                        _, _, vt = torch.linalg.svd(k_spectral_space, full_matrices=False)
                        pc1 = vt[0].reshape(1, -1)
                        projections = pc1 @ k_spectral_space.T
                        scores = torch.linalg.norm(projections, dim=0)

                    elif spectral_signature_args['scoring_fn'] == 'scaled_pca':
                        k_std = k_spectral_space.std(dim=0, keepdim=True, unbiased=False)
                        k_spectral_space = k_spectral_space / k_std
                        _, _, vt = torch.linalg.svd(k_spectral_space, full_matrices=False)
                        pc1 = vt[0].reshape(1, -1)
                        projections = pc1 @ k_spectral_space.T
                        scores = torch.linalg.norm(projections, dim=0)
                    
                    else:
                        raise NotImplementedError

                    all_scores_in_block[curr_y == k] = scores

                # TODO: delete
                noise_scale = 2 * N_drop * (1 / len(all_scores_in_block)) / (0.5 / n_epochs)
                noise = np.random.laplace(loc=0.0, scale=noise_scale, size=all_scores_in_block.shape)
                noise_tensor = torch.from_numpy(noise).to(all_scores_in_block.dtype).to(all_scores_in_block.device)
                all_scores_in_block = all_scores_in_block + noise_tensor
                all_scores_in_block_idx = torch.argsort(all_scores_in_block, descending=True)

                if len(y) - 1 in idx_block[block_drop_mask == 0]:
                    canary_rank = torch.where(all_scores_in_block_idx == (len(all_scores_in_block_idx) - 1))[0]

                    if spectral_signature_args['store_canary_rank'] is not None:
                        print('Canary Rank:', canary_rank)
                        if not canary_dropped and canary_rank < N_drop:
                            print('Canary Would Be Dropped')
                            # Plot the spectral space on the iteration where the canary is dropped
                            _, _, vt = torch.linalg.svd(k_spectral_space, full_matrices=False)
                            projections = k_spectral_space @ vt[:2].T
                            plt.clf()
                            plt.figure()
                            plt.scatter(projections[:, 0].cpu().numpy(), projections[:, 1].cpu().numpy(), alpha=0.6, color='blue')
                            plt.scatter(projections[-1, 0].cpu().numpy(), projections[-1, 1].cpu().numpy(), color='red', label='Canary')               
                            plt.xlabel('PC1')
                            plt.ylabel('PC2')
                            plt.title('Projection onto First 2 Components of Spectral Space')
                            plt.legend()
                            plt.grid(True)
                            plt.savefig(f'{out}/spectral_sig_viz.png')
                            # Plot the score distribution on the iteration where the canary is dropped
                            plt.clf()
                            plt.figure()
                            plt.plot(sorted(all_scores_in_block.cpu().numpy()), color='blue')
                            plt.plot(len(all_scores_in_block) - N_drop - 1, sorted(all_scores_in_block)[-1 * N_drop - 1].item(), marker='o', color='green', markersize=10, label='Smallest Dropped Score')
                            plt.plot(len(all_scores_in_block) - 1, all_scores_in_block[-1].item(), marker='o', color='red', markersize=10, label='Canary Score')

                            plt.title("Sorted Scores (Outliers will pop)")
                            plt.yscale('log')
                            plt.ylabel("Score")
                            plt.legend()
                            plt.grid(True)
                            plt.savefig(f'{out}/score_dist.png')

                            spectral_signature_args['canary_dropped'] = True

                        # Plot k_spectral_space in 2D
                        spectral_signature_args['store_canary_rank'].append(canary_rank.item())

                if use_defense != '':
                    drop_mask[idx_block[block_drop_mask == 0][all_scores_in_block_idx[:N_drop]]] = 1
                    for idx in all_scores_in_block_idx[:N_drop]:
                        for name, _ in model.named_parameters():
                            if max_grad_norm is not None:
                                clipping_factor = 1 / max(1, ps_grad_norms_data['before'][idx].item() / max_grad_norm)
                            else: clipping_factor = 1
                            if use_defense == 'drop':
                                accum_grad[name] -= clipping_factor * ps_grads[name][idx]
                            elif use_defense == 'grad_ascent':
                                accum_grad[name] -= 2 * clipping_factor * ps_grads[name][idx]
                            else:
                                raise NotImplementedError

    return accum_grad




def _global_clip_and_accum_grads(model, X, y, optimizer, criterion, max_grad_norm, block_size=1024, use_defense=False, spectral_signature_args=None):
    """Clip and accumulate gradients in blocks of samples to conserve gpu space"""

    N_drop = int(0.05 * len(y) // (100 * (len(y) // block_size)))

    find_outliers = False
    drop_mask = torch.zeros_like(y)

    if spectral_signature_args:
        drop_mask = spectral_signature_args['drop_mask']
        find_outliers = True


    accum_grad = None

    for k in range(0, 10):
        curr_X, curr_y = X[(y == k) & (drop_mask == 0)], y[(y == k) & (drop_mask == 0)]
        optimizer.zero_grad()

        ps_grads = get_per_sample_grads(model, curr_X, curr_y, criterion)
        ps_grads_clipped, ps_grad_norms_data = clip_per_sample_grads(ps_grads, max_grad_norm)

        with torch.no_grad():
            accum_grad_block = {name: grad.sum(dim=0) for name, grad in ps_grads_clipped.items()}
        
        del ps_grads_clipped
        torch.cuda.empty_cache()

        # accum grads for all blocks
        if accum_grad is None:
            accum_grad = accum_grad_block
        else:
            with torch.no_grad():
                for name, curr_grad in accum_grad_block.items():
                    accum_grad[name] = accum_grad[name] + curr_grad
        ###################################################

    #     if find_outliers:
    #         with torch.no_grad():

    #             k_spectral_space = None
    #             last_layer_name = list(model.net.named_modules())[-1][0]
    #             last_w_name = 'net.' + last_layer_name + '.weight'
    #             last_b_name = 'net.' + last_layer_name + '.bias'

    #             w = torch.vstack([
    #                 curr_grad[k] for curr_grad in ps_grads[last_w_name]
    #             ])

    #             b = torch.vstack([
    #                 curr_grad[k] for curr_grad in ps_grads[last_b_name]
    #             ])

    #             k_spectral_space = torch.cat((w, b), dim=1)
     
    #             # Center last layer gradients
    #             k_spectral_space = k_spectral_space - k_spectral_space.mean(dim=0, keepdim=True)
                        
    #             if spectral_signature_args['scoring_fn'] == 'whitened_norm':
    #                 W = whiten(k_spectral_space)
    #                 k_spectral_space = k_spectral_space @ W

    #             if spectral_signature_args['scoring_fn'] == 'pca':
    #                 _, _, vt = torch.linalg.svd(k_spectral_space, full_matrices=False)
    #                 pc1 = vt[0].reshape(1, -1)
    #                 projections = pc1 @ k_spectral_space.T
    #                 scores = torch.linalg.norm(projections, dim=0)

    #             elif spectral_signature_args['scoring_fn'] in ['norm', 'whitened_norm']:
    #                 # # TODO: remove this
    #                 # k_std = k_spectral_space.std(dim=0, keepdim=True, unbiased=False)
    #                 # k_spectral_space = k_spectral_space / k_std

    #                 scores = torch.linalg.norm(k_spectral_space, dim=1)

    #             all_scores_in_block_idx = torch.argsort(scores, descending=True)

    #             if drop_mask[-1] == 0 and k == y[-1]:
    #                 canary_rank = torch.where(all_scores_in_block_idx == (len(all_scores_in_block_idx) - 1))[0]

    #                 if spectral_signature_args['store_canary_rank'] is not None:
    #                     print('Canary Rank:', canary_rank)
    #                     spectral_signature_args['store_canary_rank'].append(canary_rank.item())

    #             if use_defense != '':
    #                 drop_mask[torch.where((y == k) & (drop_mask == 0))[0][all_scores_in_block_idx[:N_drop]]] = 1
    #                 for idx in all_scores_in_block_idx[:5]:
    #                     for name, _ in model.named_parameters():
    #                         clipping_factor = 1 / max(1, ps_grad_norms_data['before'][idx].item() / max_grad_norm)
    #                         if use_defense == 'drop':
    #                             accum_grad[name] -= clipping_factor * ps_grads[name][idx]
    #                         elif use_defense == 'grad_ascent':
    #                             accum_grad[name] -= 2 * clipping_factor * ps_grads[name][idx]
    #                         elif use_defense == 'adaptive_clipping':
    #                             pass

    return accum_grad