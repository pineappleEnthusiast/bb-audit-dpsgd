"""
Utility functions to execute DP-SGD
"""
import torch
import numpy as np
from torch.func import functional_call, vmap, grad
import matplotlib.pyplot as plt
import threading
import copy


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

    return ps_grads_clipped, { 'before': ps_grad_norms.cpu().numpy(), 'after': ps_grad_norms_clipped.cpu().numpy() }


def clip_and_accum_grads_block(model, X, y, optimizer, criterion, max_grad_norm, device='cuda'):
    """Clip and accumulate gradients of a single block of samples using multiple GPUs"""
    optimizer.zero_grad()

    if len(X) == 0:
        # empty dataset
        ps_grads = { name: torch.zeros_like(param).unsqueeze(dim=0) for name, param in model.named_parameters() }
    else:
        # Move data to appropriate device
        X = X.to(device)
        y = y.to(device)
        
        # Get per-sample gradients
        ps_grads = get_per_sample_grads(model, X, y, criterion)

    ps_grad_norms_data = { 'before': np.array([]), 'after': np.array([]) }
    if max_grad_norm is not None:
        # clip per-sample gradients
        # ps_grads_clipped, ps_grad_norms_data = ps_grads, ps_grad_norms_data
        ps_grads_clipped, ps_grad_norms_data = clip_per_sample_grads(ps_grads, max_grad_norm)
    else:
        ps_grads_clipped = ps_grads

    with torch.no_grad():
        accum_grad_block = {name: grad.sum(dim=0) for name, grad in ps_grads_clipped.items()}

    last_layer_name = list(model.net.named_modules())[-1][0]
    last_w_name = 'net.' + last_layer_name + '.weight'
    last_b_name = 'net.' + last_layer_name + '.bias'

    # Compute flattened norm across all param grads
    per_sample_flat_grads = torch.cat([g.view(g.shape[0], -1) for g in ps_grads.values()], dim=1)
    all_norms = torch.zeros_like(y, dtype=torch.float32)
    # for k in range(10):
    #     k_last_layer_grads = per_sample_flat_grads[y == k]
    #     centered_k_last_layer_grads = k_last_layer_grads - k_last_layer_grads.mean(dim=0, keepdim=True)
    #     # take norm of each class
    #     centered_k_last_layer_norms = centered_k_last_layer_grads.norm(float('inf'), dim=1)
    #     all_norms[y == k] = centered_k_last_layer_norms

    # all_norms = (per_sample_flat_grads - per_sample_flat_grads.mean(dim=0, keepdim=True)).norm(float('inf'), dim=1)
    
    # Compute last layer norms at true class


    # # Compute flattened last layer norms
    # flat_last_weights = ps_grads[last_w_name].flatten(start_dim=1)
    # last_biases = ps_grads[last_b_name]
    # last_layer_grads = torch.cat((flat_last_weights, last_biases), dim=1)
    
    # all_norms = torch.zeros_like(y, dtype=torch.float32)
    # for k in range(10):
    #     # center each class
    #     k_last_layer_grads = last_layer_grads[y == k]
    #     centered_k_last_layer_grads = k_last_layer_grads - k_last_layer_grads.mean(dim=0, keepdim=True)
    #     # take norm of each class
    #     centered_k_last_layer_norms = centered_k_last_layer_grads.norm(2, dim=1)
    #     all_norms[y == k] = centered_k_last_layer_norms

    last_layer_norms = all_norms.cpu().numpy()

    # Compute embedding norms
    # Cosine similarity with PC1
    # per_sample_flat_grads = torch.cat([g.view(g.shape[0], -1) for g in ps_grads.values()], dim=1)
    # X = per_sample_flat_grads
    # X_norm = X / (X.norm(dim=1, keepdim=True))
    # X_centered = X_norm - X_norm.mean(dim=0, keepdim=True)
    # U, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)
    # PC1 = Vh[0] # .reshape(1, -1)
    # cosine_sims = X_centered @ PC1
    # cosine_sims = None
    # Cosine similarity with whitened PC1
    
    return accum_grad_block, ps_grad_norms_data, last_layer_norms, None


def clip_and_accum_grads(model, X, y, optimizer, criterion, max_grad_norm, block_size=1024, drop_mask=None, device='cuda', target_in_batch=False, target_idx_in_batch=None, original_indices=None):
    """Clip and accumulate gradients in blocks of samples using multiple GPUs"""
    # X and y already contain only non-dropped indices
    # original_indices contains the original indices of these samples

    # split samples into blocks by index
    idx_blocks = torch.split(torch.from_numpy(np.arange(len(X))), block_size)

    # Split blocks among available GPUs
    if torch.cuda.device_count() > 1:
        # Get number of GPUs and blocks
        n_gpus = torch.cuda.device_count()
        n_blocks = len(idx_blocks)
        
        # Split blocks evenly among GPUs
        blocks_per_gpu = n_blocks // n_gpus
        gpu_blocks = []
        gpu_models = []
        gpu_optimizers = []
        
        # Create models and optimizers for each GPU
        for gpu_id in range(n_gpus):
            start_idx = gpu_id * blocks_per_gpu
            end_idx = (gpu_id + 1) * blocks_per_gpu if gpu_id < n_gpus - 1 else n_blocks
            gpu_blocks.append(idx_blocks[start_idx:end_idx])
            
            # Create model and optimizer for this GPU
            model_gpu = copy.deepcopy(model).to(f'cuda:{gpu_id}')
            gpu_models.append(model_gpu)
            gpu_optimizers.append(torch.optim.SGD(model_gpu.parameters(), lr=optimizer.param_groups[0]['lr']))

        def process_blocks(gpu_id, blocks, model_gpu, optimizer_gpu):
            accum_grad = None
            scores = []
            for idx_block in blocks:
                curr_X, curr_y = X[idx_block].to(f'cuda:{gpu_id}'), y[idx_block].to(f'cuda:{gpu_id}')
                accum_grad_block, _, curr_last_layer_norms, _ = clip_and_accum_grads_block(
                    model_gpu, curr_X, curr_y, optimizer_gpu, criterion, max_grad_norm, device=f'cuda:{gpu_id}'
                )
                if accum_grad is None:
                    accum_grad = accum_grad_block
                else:
                    accum_grad = {name: accum_grad[name] + accum_grad_block[name] for name in accum_grad}
                scores.extend(curr_last_layer_norms)
            return accum_grad, scores

        def thread_func(gpu_id, blocks, model_gpu, optimizer_gpu, results):
            accum_grad, scores = process_blocks(gpu_id, blocks, model_gpu, optimizer_gpu)
            results[gpu_id] = (accum_grad, scores)

        results = {}
        threads = []
        # Create threads for each GPU
        for gpu_id in range(n_gpus):
            thread = threading.Thread(
                target=thread_func, 
                args=(gpu_id, gpu_blocks[gpu_id], gpu_models[gpu_id], gpu_optimizers[gpu_id], results)
            )
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

        device = 'cuda:0'
        # Combine results from all GPUs
        accum_grad = {}
        scores = []
        for gpu_id in range(n_gpus):
            gpu_accum_grad, gpu_scores = results[gpu_id]
            if not accum_grad:
                accum_grad = {name: gpu_accum_grad[name].to(device) for name in gpu_accum_grad}
            else:
                for name in accum_grad:
                    accum_grad[name] = accum_grad[name] + gpu_accum_grad[name].to(device)
            scores.extend(gpu_scores)

        scores = np.array(scores)

        # Copy gradients back to original model
        for name, param in model.named_parameters():
            param.grad = accum_grad[name].to(device)

    else:
        # Process blocks in serial order
        accum_grad = None
        scores = []

        for idx_block in idx_blocks:
            # get a single block of samples
            curr_X, curr_y = X[idx_block], y[idx_block]
        
            # accum grads for this single block
            accum_grad_block, curr_ps_grad_norms_data, curr_last_layer_norms, curr_cosine_sims = clip_and_accum_grads_block(model, curr_X, curr_y, optimizer, criterion, max_grad_norm, device=device)

            # Store before norms in scores
            # scores.append(curr_ps_grad_norms_data['before'])
            scores.append(curr_last_layer_norms)
            # scores.append(curr_cosine_sims)

            # accum grads for all blocks
            if accum_grad is None:
                accum_grad = accum_grad_block
            else:
                with torch.no_grad():
                    for name, curr_grad in accum_grad_block.items():
                        accum_grad[name] = accum_grad[name] + curr_grad

        scores = np.concatenate(scores)
    
    
    # Print indices of top 5 scores
    # k = 5
    # topk_idx = np.argpartition(-scores, k)[:k]
    # print('Top Across D:', topk_idx)
    # print('Max vs Canary Score Across D', max(scores), scores[-1], min(scores))
    # if len(scores) - 1 in topk_idx:
    #     print('CANARY GETS THROWN OUT full')
    # canary_class_scores = scores[y.cpu().numpy() == y[-1].cpu().numpy()]
    # print('Canary index in D[k]', len(canary_class_scores) - 1)
    # topk_idx_class = np.argpartition(-canary_class_scores, k)[:k]
    # print('Top Across D[k]', topk_idx_class)
    # if len(canary_class_scores) - 1 in topk_idx_class:
    #     print('CANARY GETS THROWN OUT')
    # print('Max vs Canary Score Across D[k]', max(canary_class_scores), canary_class_scores[-1], min(canary_class_scores))
    
    # Privatize scores
    # Per-class:
        # Clip the per-sample gradients
        # Take the mean of the clipped gradients
        # Recenter the unclipped gradients with this clipped mean
        # Take the norm of the unclipped but centered gradients
        # Add Laplace noise
        # Choose top-k
    # Print indices of top 5 scores

    # global_indices = torch.arange(start=0, end=len(drop_mask), step=1, device=y.device)
    # active_global_indices = global_indices[drop_mask == 0]
    # global_indices_to_filter = active_global_indices[topk_idx]
    # drop_mask[global_indices_to_filter] = 1

    # Recompute gradients for top 5 indices
    # X_filter, y_filter = X[topk_idx], y[topk_idx]
    # filter_accum_grad_block, _, _, _ = clip_and_accum_grads_block(model, X_filter, y_filter, optimizer, criterion, max_grad_norm)

    # subtract from accum grad
    # with torch.no_grad():
    #     for name, curr_grad in filter_accum_grad_block.items():
    #         accum_grad[name] = accum_grad[name] - curr_grad
    
    return accum_grad, drop_mask



# TODO: switch back from ps_grads_clipped, get rid of noised gradients
def ___local_clip_and_accum_grads(model, X, y, optimizer, criterion, max_grad_norm, block_size=1024, use_defense=False, spectral_signature_args=None):
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

