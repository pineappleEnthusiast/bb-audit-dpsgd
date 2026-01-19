
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import argparse
import copy
import math
import random
from utils.data import load_data
from torch.func import functional_call, vmap, grad
from tqdm import tqdm

# ==========================================
# Step 1: Metasmooth Surrogate Model
# ==========================================

class Mul(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight
    def forward(self, x):
        return x * self.weight

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Residual(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
    def forward(self, x):
        return x + self.module(x)

def metasmooth_conv_bn(channels_in, channels_out, kernel_size=3, stride=1, padding=1, groups=1):
    return nn.Sequential(
        nn.Conv2d(channels_in, channels_out, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
        nn.BatchNorm2d(channels_out, track_running_stats=False),
        nn.GELU()
    )

class MetaSmoothResNet9(nn.Module):
    def __init__(self, num_classes=10, width_mult=2.0):
        super().__init__()
        c = int(64 * width_mult)
        
        self.prep = metasmooth_conv_bn(3, c)
        
        self.layer1 = nn.Sequential(
            metasmooth_conv_bn(c, c*2),
            nn.AvgPool2d(2)
        )
        
        self.res1 = Residual(nn.Sequential(
            metasmooth_conv_bn(c*2, c*2),
            metasmooth_conv_bn(c*2, c*2)
        ))
        
        self.layer2 = nn.Sequential(
            metasmooth_conv_bn(c*2, c*4),
            nn.AvgPool2d(2)
        )
        
        self.res2 = Residual(nn.Sequential(
            metasmooth_conv_bn(c*4, c*4),
            metasmooth_conv_bn(c*4, c*4)
        ))
        
        self.layer3 = nn.Sequential(
            metasmooth_conv_bn(c*4, c*8),
            nn.AvgPool2d(4)
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
            nn.Linear(c*8, num_classes),
            Mul(0.125)
        )

    def forward(self, x):
        out = self.prep(x)
        out = self.layer1(out)
        out = self.res1(out)
        out = self.layer2(out)
        out = self.res2(out)
        out = self.layer3(out)
        out = self.classifier(out)
        return out

# ==========================================
# Step 2: REPLAY Data Structure (Lazy k-ary Tree)
# ==========================================

class LazyTree:
    """
    Implements a lazy checkpointing strategy. 
    To support recreating any state t with minimal memory.
    Simple strategy: Store state every S steps.
    To get t: Find nearest checkpoint t_prev <= t, and simulate forward.
    """
    def __init__(self, period_sqrt_T=True, forced_period=None):
        self.checkpoints = {}
        self.period_sqrt_T = period_sqrt_T
        self.period = forced_period
        
    def setup(self, total_steps, initial_state):
        if self.period is None:
            self.period = int(math.sqrt(total_steps)) + 1
        self.checkpoints[0] = copy.deepcopy(initial_state)
        
    def maybe_save(self, step, state):
        if step % self.period == 0:
            self.checkpoints[step] = copy.deepcopy(state)
            
    def get_leaf(self, target_step, simulator_fn, batch_provider_fn):
        """
        Recover state at target_step.
        simulator_fn: (state, batch) -> next_state
        batch_provider_fn: (step) -> batch
        """
        # 1. Find nearest previous checkpoint
        # keys are steps. find max key <= target_step
        available_steps = [s for s in self.checkpoints.keys() if s <= target_step]
        start_step = max(available_steps)
        current_state = copy.deepcopy(self.checkpoints[start_step])
        
        # 2. Simulate forward
        for t in range(start_step, target_step):
            batch = batch_provider_fn(t)
            # Make sure we don't build graph here during pure replay forward
            with torch.no_grad():
                current_state = simulator_fn(current_state, batch)
                
        return current_state

    def free_memory(self):
        self.checkpoints.clear()

# ==========================================
# Step 3, 4, 5: Optimization & Metagradient
# ==========================================

def get_params_buffers(model):
    # Returns purely functional params/buffers dicts
    return dict(model.named_parameters()), dict(model.named_buffers())

def functional_loss(params, buffers, x, y, model, num_classes=10):
    # x: (B, C, H, W)
    # y: (B)
    logits = functional_call(model, (params, buffers), (x.unsqueeze(0),)) # func_call expects batched input? 
    # Actually functional_call expects (args, kwargs). If x is a batch, we pass (x,).
    # But usually vmap adds the batch dimension.
    # If we are NOT using vmap here (just normal functional call):
    # The 'model' forward expects 'x'.
    return F.cross_entropy(logits, y.unsqueeze(0))

def compute_per_sample_grads(params, buffers, sample, target, model):
    # wrapper for vmap
    # sample: (C, H, W), target: ()
    # returns: grads w.r.t params
    
    def loss_fn(p, b, x, y):
        # Forward pass for single sample
        # Add batch dim for model
        out = functional_call(model, (p, b), (x.unsqueeze(0),))
        return F.cross_entropy(out, y.unsqueeze(0))

    grad_fn = grad(loss_fn)
    # We want gradients w.r.t params only
    return grad_fn(params, buffers, sample, target)

def differentiable_sgd_step(params, buffers, batch_x, batch_y, model, lr, max_grad_norm, momentum_state=None, momentum=0.9):
    """
    Performs one step of DP-SGD in a differentiable manner.
    Input:
        params: dict of tensors (requires_grad might be True if we are inside re-simulation)
        batch_x: (B, C, H, W). If it contains canary, it should be connected to canary tensor graph.
    Returns:
        next_params: dict
        next_momentum: dict
    """
    B = batch_x.shape[0]
    
    # 1. Compute Per-Sample Gradients
    # Use vmap to efficiently compute per-sample grads
    # 'func_model' is the stateless model instance
    
    # Define per-sample gradient function
    def compute_grad_single(p, b, x, y):
        out = functional_call(model, (p, b), (x.unsqueeze(0),))
        loss = F.cross_entropy(out, y.unsqueeze(0))
        return loss

    # Compute grads w.r.t params. args: (params, buffers, x, y)
    # vmap over x and y (dims 0), params/buffers are shared (None)
    grad_fn = grad(compute_grad_single)
    per_sample_grads = vmap(grad_fn, in_dims=(None, None, 0, 0))(params, buffers, batch_x, batch_y)
    
    # 2. Clip Gradients
    # Flatten all grads for each sample to compute norm
    flat_grads = []
    # Sort keys to ensure deterministic order
    keys = sorted(params.keys())
    for k in keys:
        g = per_sample_grads[k]
        flat_grads.append(g.reshape(B, -1))
    
    flat_grads_cat = torch.cat(flat_grads, dim=1)
    norms = torch.norm(flat_grads_cat, dim=1)
    
    scales = torch.clamp(max_grad_norm / (norms + 1e-6), max=1.0) # (B,)
    
    # 3. Average (DP-SGD)
    # Sum clipped grads
    summed_grads = {}
    for k in keys:
        g = per_sample_grads[k] # (B, ...)
        # Reshape scale for broadcasting
        # (B, 1, 1, ...)
        view_shape = [B] + [1] * (g.ndim - 1)
        s = scales.view(*view_shape)
        
        g_clipped = g * s
        summed_grads[k] = g_clipped.sum(dim=0) # Sum over batch
        
    # 4. Update with Momentum
    next_params = {}
    next_momentum = {}
    
    if momentum_state is None:
        # Initialize zero momentum
        momentum_state = {k: torch.zeros_like(v) for k, v in params.items()}
        
    for k in keys:
        # Standard SGD with Momentum:
        # v_{t+1} = mu * v_t + g
        # p_{t+1} = p_t - lr * v_{t+1}
        # Note: DP-SGD usually adds noise to 'summed_grads' before dividing by B.
        # We assume 0 noise for attack generation (or mean field).
        
        g_avg = summed_grads[k] / B
        
        v_next = momentum * momentum_state[k] + g_avg
        p_next = params[k] - lr * v_next
        
        next_params[k] = p_next
        next_momentum[k] = v_next
        
    return next_params, buffers, next_momentum

def get_batch_indices(t, batch_size, total_samples, indices_permutation):
    # Standard epoch-based shuffling or infinite streaming
    # We will just cycle through permutation
    start = (t * batch_size) % total_samples
    end = start + batch_size
    if end > total_samples:
        # Wrap around
        idx = np.concatenate([indices_permutation[start:], indices_permutation[:end-total_samples]])
    else:
        idx = indices_permutation[start:end]
    return idx

def metagradient_attack(
    device='cuda',
    canary_count=1000,
    n_train_data=49000,
    meta_iterations=100,
    inner_steps=200, # T
    batch_size=250,
    lr=0.1,
    momentum=0.9,
    max_grad_norm=1.0,
    outer_lr=0.1
):
    # 1. Setup Data
    print("Loading Data...")
    root = './'
    X_mnist, y_mnist, _ = load_data('mnist', n_df=None, root=root, split='train')
    # Take subset
    X_train = X_mnist[:n_train_data].to(device)
    y_train = y_mnist[:n_train_data].to(device)
    
    # 2. Initialize Canaries (Random Noise or from Data)
    # Initialize as parameter
    # Random normal initialization
    canaries = torch.randn(canary_count, 3, 32, 32, device=device, requires_grad=True)
    # MNIST is 1 channel, but ResNet9 is 3 channel. Expand/Repeat or just user 1 channel?
    # Provided dpsgd code manages 1 channel MNIST? 
    # Usually we use 3 channels for ResNet.
    # Check X_train shape
    if X_train.shape[1] == 1:
        # Repeat to 3 channels for ResNet9 compatibility if needed, 
        # OR modify ResNet9 to take 1 channel. 
        # Source code implies standard models, likely 3 channel.
        # Let's check X_train dimensions from load_data.
        # MNIST is (N, 1, 28, 28). We need to resize to 32x32 maybe? 
        # CIFAR is 3x32x32.
        # The prompt says: "compatible with the MNIST dataset we load in this repository".
        # dpsgd.py uses models.lstm etc.
        # Let's assume we need to adapt MNIST to 3x32x32 or modify model.
        # Safest: Modify MetaSmoothResNet9 to take 1 channel if dataset is MNIST.
        # BUT Plan says "Differentiable Surrogate Model... ResNet-9".
        # Let's check input channels in MetaSmoothResNet9.__init__: self.prep = metasmooth_conv_bn(3, c)
        # So it expects 3.
        # I will resize/repeat MNIST to 3x32x32.
        X_train_resized = F.interpolate(X_train, size=(32, 32), mode='bilinear')
        X_train_final = X_train_resized.repeat(1, 3, 1, 1)
        # Canaries also 3x32x32
    else:
        X_train_final = X_train
        
    y_canaries = torch.randint(0, 10, (canary_count,), device=device) # Random labels? Or specific?
    # Usually canaries have target labels. Let's stick to random for generic attack.
    
    print(f"Data Shape: {X_train_final.shape}")
    
    # Optimizer for canaries
    # We will manually update canaries to clip them
    
    # 3. Main Loop
    history = []
    
    for meta_iter in range(meta_iterations):
        print(f"=== Meta Iteration {meta_iter+1}/{meta_iterations} ===")
        
        # 3a. Partition Canaries
        perm = torch.randperm(canary_count)
        split = canary_count // 2
        idx_in = perm[:split]
        idx_out = perm[split:]
        
        mask_in = torch.zeros(canary_count, dtype=torch.bool, device=device)
        mask_in[idx_in] = True
        
        # Combine Data: D + C_in
        # We don't physically concat tensors to avoid breaking graph for C.
        # We handle this in batch sampling.
        
        total_train_indices = np.array(list(range(n_train_data + split))) # 0..N-1 are MNIST, N..N+Split-1 are C_in
        np.random.shuffle(total_train_indices) # Shuffle once per epoch/training run
        
        # 3b. REPLAY Training (Forward)
        model = MetaSmoothResNet9(num_classes=10).to(device)
        model.train()
        
        # Initial Weights
        params = {k: v.detach().clone().requires_grad_(True) for k, v in model.named_parameters()}
        buffers = {k: v.detach().clone() for k, v in model.named_buffers()}
        momentum_state = {k: torch.zeros_like(v) for k, v in params.items()}
        
        # Package state
        state = (params, buffers, momentum_state)
        
        lazy_tree = LazyTree()
        lazy_tree.setup(inner_steps, state)
        
        # Batch Provider helper
        def get_batch(step):
            # Deterministic batch for step t (given the shuffle above)
            indices = get_batch_indices(step, batch_size, len(total_train_indices), total_train_indices)
            
            # Map indices to data
            # indices < n_train_data => MNIST
            # indices >= n_train_data => C_in (mapped)
            
            # Separate logic
            is_mnist = indices < n_train_data
            mnist_idx = indices[is_mnist]
            
            canary_relative_idx = indices[~is_mnist] - n_train_data
            canary_abs_idx = idx_in[canary_relative_idx] # Map back to global canary list
            
            x_m = X_train_final[mnist_idx]
            y_m = y_train[mnist_idx]
            
            # For forward pass, we detach canaries! We only need graph in backward replay.
            # Wait, LazyTree instructions: "Forward... Do not detach gradients, but do not cache the full graph."
            # If we don't detach, we build a graph.
            # "Do not cache the full graph" means we can't keep all steps in memory.
            # If we keep graph attached, we run OOM.
            # So we MUST detach for memory, but the prompt says "Do not detach gradients".
            # This is contradictory unless they mean "Do not detach w.r.t canaries but use checkpointing".
            # BUT checkpointing is manual.
            # Standard Replay pattern:
            # Pass 1 (Trace): Run forward with detach(). Save checkpoints.
            # Pass 2 (Diff): Run backward. Re-run forward segments WITH graph.
            
            # So here in "Forward", we run with detach (no_grad).
            x_c = canaries[canary_abs_idx].detach() 
            y_c = y_canaries[canary_abs_idx]
            
            if len(x_m) > 0 and len(x_c) > 0:
                x_batch = torch.cat([x_m, x_c], dim=0)
                y_batch = torch.cat([y_m, y_c], dim=0)
            elif len(x_m) > 0:
                x_batch = x_m
                y_batch = y_m
            else:
                x_batch = x_c
                y_batch = y_c
                
            return x_batch, y_batch, canary_abs_idx

        # Forward Training Loop
        for t in tqdm(range(inner_steps), desc="Forward Training"):
            x, y, _ = get_batch(t)
            
            # Step
            curr_params, curr_buffers, curr_mom = state
            
            # We treat everything as detached here
            with torch.no_grad():
                new_params, new_buffers, new_mom = differentiable_sgd_step(
                    curr_params, curr_buffers, x, y, model, lr, max_grad_norm, curr_mom, momentum
                )
            
            state = (new_params, new_buffers, new_mom)
            lazy_tree.maybe_save(t+1, state)
        
        final_params = state[0]
        final_buffers = state[1]
        
        # 3c. Compute Gradient at T (d_phi / d_wT)
        # Objective: sum( 1_in * Loss - 1_out * Loss )
        # We need grads w.r.t final weights.
        
        # To get dPhdw_T, we compute phi(w_T) and backprop.
        # But w_T is detached.
        # We need to temporarily attach it or just use functional grad.
        
        # Create a new graph root at final_params
        w_T = {k: v.detach().requires_grad_(True) for k, v in final_params.items()}
        
        # Compute Phi
        # We evaluate on ALL canaries
        # In batches if too many? 1000 is small enough for 1 batch usually (32x32 imgs).
        # We'll do it in chunks to be safe.
        
        total_obj = 0
        w_T_grads = {k: torch.zeros_like(v) for k, v in w_T.items()}
        
        # Evaluate All Canaries
        # We need Loss(w_T, z_i).
        # Mask: +1 for in, -1 for out.
        signs = torch.ones(canary_count, device=device)
        signs[idx_out] = -1.0
        # signs[idx_in] = 1.0 (already 1)
        
        def compute_weighted_loss(p, b, x, y, s):
            out = functional_call(model, (p, b), (x.unsqueeze(0),))
            l = F.cross_entropy(out, y.unsqueeze(0))
            return l * s

        # Batched Grad of Phi
        # vmap over canaries
        # args: (w_T, buffers, canaries, y_canaries, signs)
        grad_phi_fn = grad(compute_weighted_loss)
        # We want gradients w.r.t w_T (arg 0)
        # vmap dim: arg0=None, arg1=None, arg2=0, arg3=0, arg4=0
        
        batch_eval = 250
        canary_grad_accum = torch.zeros_like(canaries) # dPhi/dCanary (accumulated during backtracking)
        
        # Here we only compute dPhi/dw_T.
        # Canaries are constants for this derivative w.r.t w_T.
        for i in range(0, canary_count, batch_eval):
            end = min(i + batch_eval, canary_count)
            c_batch = canaries[i:end].detach() # Detached! We define objective on w_T given C.
            y_batch = y_canaries[i:end]
            s_batch = signs[i:end]
            
            # Compute per-sample grads of w_T weighted by sign
            grads_per_sample = vmap(grad_phi_fn, in_dims=(None, None, 0, 0, 0))(
                w_T, final_buffers, c_batch, y_batch, s_batch
            )
            
            # valid gradients are summed
            with torch.no_grad():
                for k in w_T_grads:
                    w_T_grads[k] += grads_per_sample[k].sum(dim=0)
                    
        # current_grad = dPhi/dw_T
        current_w_grads = w_T_grads
        
        # 3d. Backward Pass (Replay)
        # Iterate T-1 down to 0
        
        for t in tqdm(reversed(range(inner_steps)), desc="Backward Replay"):
            # 1. Recover w_t
            # We need w_t, buffer_t, mom_t
            # We can use LazyTree to get state at t
            
            def sim_fn(s, b): # wrapper for lazy tree
                # s is state, b is (x, y, _)
                p, buf, m = s
                # Simulator must match forward pass exactly (no grad)
                new_p, new_buf, new_m = differentiable_sgd_step(p, buf, b[0], b[1], model, lr, max_grad_norm, m, momentum)
                return (new_p, new_buf, new_m)
                
            prev_state_detached = lazy_tree.get_leaf(t, sim_fn, get_batch)
            params_t_det, buffers_t_det, mom_t_det = prev_state_detached
            
            # 2. Re-run Step t -> t+1 WITH GRAPHS
            # Enable gradients on params_t and canaries (if present)
            params_t = {k: v.detach().requires_grad_(True) for k, v in params_t_det.items()}
            # Buffers usually constant?
            buffers_t = buffers_t_det
            mom_t = {k: v.detach().requires_grad_(True) for k, v in mom_t_det.items()} # Differentiation through momentum?
            # Yes, standard metagradients differentiate through moment updates.
            
            # Get data
            x_b, y_b, c_abs_idx = get_batch(t)
            
            # Re-construct batch with attachment to Canaries
            # Split x_b back into MNIST and Canaries
            # c_abs_idx tells us which canaries were used
            if len(c_abs_idx) > 0:
                # We need to construct x_input that is connected to 'canaries' tensor graph
                # x_b contains: [mnist_samples..., canary_samples...]
                # We rebuild it.
                
                # Identify mnist part
                n_c = len(c_abs_idx)
                n_m = x_b.shape[0] - n_c
                
                mnist_part = x_b[:n_m] # detached
                canary_part = canaries[c_abs_idx] # Attached to 'canaries'!
                
                x_input = torch.cat([mnist_part, canary_part], dim=0)
            else:
                x_input = x_b # detached
                
            # Perform Differentiable Step
            # functional_call etc supports backprop if params require grad
            next_params, _, next_mom = differentiable_sgd_step(
                params_t, buffers_t, x_input, y_b, model, lr, max_grad_norm, mom_t, momentum
            )
            
            # 3. Backprop gradients from t+1 to t
            # We have dPhi/dw_{t+1} (current_w_grads) and dPhi/dv_{t+1} (current_mom_grads?)
            # Usually only w matters, but v affects w.
            # If we track full state, we need dPhi/state_{t+1}.
            # At step T, dPhi/dv_T = 0 (Objective doesn't depend on momentum).
            
            if t == inner_steps - 1:
                # Initialize momentum grads to 0
                current_mom_grads = {k: torch.zeros_like(v) for k, v in mom_t.items()}
            
            # VJP (Vector-Jacobian Product)
            # We want: grad_inputs = (dPhi/dw_{t+1}) * (dw_{t+1}/dw_t) + ...
            # essentially: backward() on (next_params * current_w_grads + next_mom * current_mom_grads)
            
            # Helper to compute dot product scalar for backward
            dot_product = 0
            for k in next_params:
                dot_product += (next_params[k] * current_w_grads[k]).sum()
                dot_product += (next_mom[k] * current_mom_grads[k]).sum()
                
            # Compute gradients of dot_product w.r.t inputs (params_t, mom_t, x_input)
            # x_input depends on canaries. params_t depends on previous step.
            
            grads_result = torch.autograd.grad(
                dot_product, 
                list(params_t.values()) + list(mom_t.values()) + [x_input], 
                retain_graph=False # One step only
            )
            
            # Unpack
            n_p = len(params_t)
            n_m = len(mom_t)
            
            grad_params_t = dict(zip(params_t.keys(), grads_result[:n_p]))
            grad_mom_t = dict(zip(mom_t.keys(), grads_result[n_p:n_p+n_m]))
            grad_x = grads_result[-1]
            
            # Accumulate canary gradients
            if len(c_abs_idx) > 0:
                # grad_x corresponds to [mnist..., canary...]
                # Extract canary part
                n_c = len(c_abs_idx)
                n_m_samples = x_input.shape[0] - n_c
                grad_c_batch = grad_x[n_m_samples:]
                
                # Accumulate into master canary grad
                # canaries.grad is not automatically set, we accumulate manually
                # But 'canaries' is leaf. We can just modify its .grad if we want, or store separately.
                # Since we are essentially doing manual backprop, we store separately.
                if canaries.grad is None:
                    canaries.grad = torch.zeros_like(canaries)
                
                # We need to scatter add
                canaries.grad.index_add_(0, c_abs_idx, grad_c_batch)
                
            # Update current gradients for next iteration (t-1)
            current_w_grads = grad_params_t
            current_mom_grads = grad_mom_t
            
            # End of Backward Step T
            
        # 3e. Update Canaries
        # Now we have dPhi/dC accumulated in canaries.grad
        # Apply Optimizer step on Canaries
        # Objective is to MINIMIZE Phi. So we move against gradient?
        # "Maximize the loss gap". Phi has (1_in - 1_out).
        # We want In to have Low Loss, Out to have High Loss.
        # So we want to Minimize (Loss_In - Loss_Out).
        # Yes, minimize Phi.
        # So C = C - lr * grad
        
        with torch.no_grad():
            # Basic SGD update for canaries
            # Optional: Momentum for canaries separately?
            # User says "Outer Optimizer: PGD or Adam". Let's use simple PGD (SGD + Clip).
            
            grad_C = canaries.grad
            if grad_C is None:
                 grad_C = torch.zeros_like(canaries)
                 
            # Update
            canaries.data -= outer_lr * grad_C
            
            # Project (PGD)
            canaries.data.clamp_(-2.0, 2.0) # MNIST normalized range?
            # MNIST (0.1307, 0.3081). Min pixel ~ -0.42, Max ~ 2.8.
            # We'll valid range approx [-0.5, 3.0] or just clamp loosely.
            # Strictly: [ (0-mu)/sigma, (1-mu)/sigma ]
            min_val = (0.0 - 0.1307) / 0.3081
            max_val = (1.0 - 0.1307) / 0.3081
            canaries.data.clamp_(min_val, max_val)
            
            # Zero grad for next meta-step
            canaries.grad.zero_()
            
        # Save visuals or stats
        # For audit, we just need the canaries saved at the end
        
    # Save Final Canaries
    save_path = "generated_metagradient_canaries.pt"
    torch.save(canaries.detach().cpu(), save_path)
    print(f"Canaries saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--canary_count', type=int, default=1000)
    parser.add_argument('--n_train', type=int, default=49000)
    parser.add_argument('--meta_iters', type=int, default=100)
    parser.add_argument('--inner_steps', type=int, default=200) # Should be larger for real convergence
    args = parser.parse_args()
    
    metagradient_attack(
        canary_count=args.canary_count,
        n_train_data=args.n_train,
        meta_iterations=args.meta_iters,
        inner_steps=args.inner_steps
    )
