import numpy as np
import os
from utils.audit import compute_eps_lower_from_mia, compute_eps_lower_from_mia_given_t


def compute_empirical_epsilon(losses_in_file, losses_out_file, 
                              alpha=0.05, delta=1e-5, 
                              use_holdout=True, seed=0, holdout_fraction=0.75):
    import glob
    
    # Try simple loading first
    losses_in = np.load(losses_in_file)
    losses_out = np.load(losses_out_file)
    
    # Check if we have a size mismatch (indicating we need rank file aggregation)
    if len(losses_in) != len(losses_out):
        # Try to aggregate rank files
        base_dir_in = os.path.dirname(losses_in_file)
        base_name_in = os.path.basename(losses_in_file)
        rank_pattern_in = base_name_in.replace('.npy', '_rank*.npy')
        rank_files_in = sorted(glob.glob(os.path.join(base_dir_in, rank_pattern_in)))
        
        base_dir_out = os.path.dirname(losses_out_file)
        base_name_out = os.path.basename(losses_out_file)
        rank_pattern_out = base_name_out.replace('.npy', '_rank*.npy')
        rank_files_out = sorted(glob.glob(os.path.join(base_dir_out, rank_pattern_out)))
        
        if rank_files_in and rank_files_out:
            # Aggregate rank files
            all_losses_in = []
            for rank_file in rank_files_in:
                all_losses_in.extend(np.load(rank_file))
            
            all_losses_out = []
            for rank_file in rank_files_out:
                all_losses_out.extend(np.load(rank_file))
            
            losses_in = np.array(all_losses_in)
            losses_out = np.array(all_losses_out)
            
            # Use minimum length to balance
            min_len = min(len(losses_in), len(losses_out))
            losses_in = losses_in[:min_len]
            losses_out = losses_out[:min_len]
    
    n = len(losses_in)
    
    # Print mean losses for diagnostics
    print(f"  Mean losses_in: {np.mean(losses_in):.6f}")
    print(f"  Mean losses_out: {np.mean(losses_out):.6f}")
    print(f"  Loss gap (in - out): {np.mean(losses_in) - np.mean(losses_out):.6f}")

    t_losses = {'in': None, 'out': None}
    holdout_losses = {'in': None, 'out': None}
    
    if use_holdout:
        np.random.seed(seed)
        indices = np.random.permutation(n)
        
        threshold_size = int(n * (1 - holdout_fraction))
        threshold_indices = indices[:threshold_size]
        holdout_indices = indices[threshold_size:]
        
        t_losses['in'] = losses_in[threshold_indices]
        t_losses['out'] = losses_out[threshold_indices]
        holdout_losses['in'] = losses_in[holdout_indices]
        holdout_losses['out'] = losses_out[holdout_indices]
    else:
        t_losses['in'] = losses_in
        t_losses['out'] = losses_out
    
    mia_scores = np.concatenate([t_losses['in'], t_losses['out']])
    mia_labels = np.concatenate([np.ones_like(t_losses['in']), np.zeros_like(t_losses['out'])])

    print(f"  MIA scores shape: {mia_scores.shape}, labels shape: {mia_labels.shape}")
    print(f"  MIA scores range: [{np.min(mia_scores):.4f}, {np.max(mia_scores):.4f}]")
    
    max_t, emp_eps_loss = compute_eps_lower_from_mia(mia_scores, mia_labels, alpha, delta, 'GDP', n_procs=1)
    
    print(f"  Optimal threshold: {max_t:.6f}")
    
    # Calculate accuracy at this threshold
    predictions = (mia_scores > max_t).astype(int)
    accuracy = np.mean(predictions == mia_labels)
    tp = np.sum((predictions == 1) & (mia_labels == 1))
    fp = np.sum((predictions == 1) & (mia_labels == 0))
    tn = np.sum((predictions == 0) & (mia_labels == 0))
    fn = np.sum((predictions == 0) & (mia_labels == 1))
    
    print(f"  MIA Accuracy: {accuracy:.4f}")
    print(f"  TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
    print(f"  Empirical epsilon (threshold split): {emp_eps_loss:.6f}")

    if use_holdout:
        emp_eps_loss = compute_eps_lower_from_mia_given_t(np.concatenate(
            [holdout_losses['in'], holdout_losses['out']]), 
            np.concatenate([np.ones_like(holdout_losses['in']), np.zeros_like(holdout_losses['out'])]), 
            alpha, 
            delta, 
            max_t, 
            'GDP')
    
    return emp_eps_loss


def main():
    # ========== CONFIGURE THESE VALUES ==========
    alpha = 0.05
    # alpha = 0.1
    delta = 1e-5
    seed = 0
    
    # Define all experiments to process
    experiments = [
        ("MNIST Poisson No Defense", "mnist_poisson_no_defense/mnist_cnn_eps10.0"),
        ("MNIST Poisson Defense", "mnist_poisson_defense/mnist_cnn_eps10.0"),
    ]
    # ============================================
    
    
    results = []
    
    for exp_name, exp_dir in experiments:
        losses_in_file = os.path.join(exp_dir, "losses_in.npy")
        losses_out_file = os.path.join(exp_dir, "losses_out.npy")
        
        # Check if files exist
        if not os.path.exists(losses_in_file) or not os.path.exists(losses_out_file):
            results.append((exp_name, None))
            continue
        
        print(f"\n{exp_name}:")
        
        # Compute with 50% holdout
        emp_eps_50 = compute_empirical_epsilon(
            losses_in_file, losses_out_file,
            alpha=alpha, delta=delta,
            use_holdout=True, seed=seed, holdout_fraction=0.50
        )
        
        # Compute with 75% holdout
        emp_eps_75 = compute_empirical_epsilon(
            losses_in_file, losses_out_file,
            alpha=alpha, delta=delta,
            use_holdout=True, seed=seed, holdout_fraction=0.75
        )
        
        # Determine best
        if emp_eps_75 >= emp_eps_50:
            best_eps = emp_eps_75
            best_holdout = "75%"
        else:
            best_eps = emp_eps_50
            best_holdout = "50%"
        
        results.append((exp_name, best_eps, best_holdout))
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY - Empirical Epsilon (with holdout):")
    print("="*60)
    for exp_name, emp_eps, holdout in results:
        if emp_eps is not None:
            print(f"{exp_name}: {emp_eps:.4f} ({holdout} holdout)")
        else:
            print(f"{exp_name}: MISSING FILES")


if __name__ == '__main__':
    main()
