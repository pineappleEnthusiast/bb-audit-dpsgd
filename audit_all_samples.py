import numpy as np
import os
import argparse
import time
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed
from utils.audit import compute_eps_lower_from_mia, compute_eps_lower_from_mia_given_t


def compute_epsilon_for_sample(args):
    """Compute empirical epsilon for a single sample using 75% holdout."""
    sample_idx, losses_in_col, losses_out_col, alpha, delta = args
    
    # Negate losses (higher loss = more likely to be member)
    scores_in = -losses_in_col
    scores_out = -losses_out_col
    
    # Implement 75% holdout split
    n = len(scores_in)
    holdout_fraction = 0.75
    
    # Use fixed seed for reproducibility across samples
    np.random.seed(sample_idx)
    indices = np.random.permutation(n)
    
    threshold_size = int(n * (1 - holdout_fraction))
    threshold_indices = indices[:threshold_size]
    holdout_indices = indices[threshold_size:]
    
    # Split for threshold finding
    t_scores_in = scores_in[threshold_indices]
    t_scores_out = scores_out[threshold_indices]
    
    # Combine threshold split
    t_scores = np.concatenate([t_scores_in, t_scores_out])
    t_labels = np.concatenate([
        np.ones(len(t_scores_in)),
        np.zeros(len(t_scores_out))
    ])
    
    # Find optimal threshold on threshold split (n_procs=1 since we parallelize at batch level)
    max_t, _ = compute_eps_lower_from_mia(
        t_scores,
        t_labels,
        alpha,
        delta,
        method='GDP',
        n_procs=1
    )
    
    # Split for holdout evaluation
    h_scores_in = scores_in[holdout_indices]
    h_scores_out = scores_out[holdout_indices]
    
    # Evaluate on holdout split
    h_scores = np.concatenate([h_scores_in, h_scores_out])
    h_labels = np.concatenate([
        np.ones(len(h_scores_in)),
        np.zeros(len(h_scores_out))
    ])
    
    emp_eps = compute_eps_lower_from_mia_given_t(
        h_scores,
        h_labels,
        alpha,
        delta,
        max_t,
        method='GDP'
    )
    
    return sample_idx, emp_eps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing all_losses_in.npy and all_losses_out.npy')
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--delta', type=float, default=1e-5)
    parser.add_argument('--n_procs', type=int, default=None,
                        help='Number of processes (default: all CPUs)')
    parser.add_argument('--out', type=str, default=None,
                        help='Output file for per-sample epsilons (default: data_dir/per_sample_epsilons.npy)')
    args = parser.parse_args()
    
    # Load all losses
    print("Loading all_losses files...")
    all_losses_in = np.load(os.path.join(args.data_dir, 'all_losses_in.npy'), allow_pickle=True)
    all_losses_out = np.load(os.path.join(args.data_dir, 'all_losses_out.npy'), allow_pickle=True)
    
    n_models, n_samples = all_losses_in.shape
    print(f"Loaded: {n_models} models, {n_samples} samples")
    print(f"Shape: all_losses_in={all_losses_in.shape}, all_losses_out={all_losses_out.shape}")
    
    # Convert to float arrays
    all_losses_in = all_losses_in.astype(np.float32)
    all_losses_out = all_losses_out.astype(np.float32)
    
    # Prepare arguments for processing
    print(f"\nComputing empirical epsilon for {n_samples} samples...")
    task_args = [
        (i, all_losses_in[:, i], all_losses_out[:, i], args.alpha, args.delta)
        for i in range(n_samples)
    ]
    
    # Use ProcessPoolExecutor with batched submission (like audit_o1_multi_canary.py)
    n_procs = args.n_procs or cpu_count()
    print(f"Using {n_procs} parallel workers with batched submission")
    
    results = []
    batch_size = 1000  # Submit in batches to avoid memory issues
    total_samples = len(task_args)
    processed = 0
    
    start_time = time.time()
    with ProcessPoolExecutor(max_workers=n_procs) as executor:
        for batch_start in range(0, total_samples, batch_size):
            batch_end = min(batch_start + batch_size, total_samples)
            batch = task_args[batch_start:batch_end]
            
            print(f"Submitting batch {batch_start//batch_size + 1}/{(total_samples + batch_size - 1)//batch_size} ({len(batch)} samples)...")
            futures = [executor.submit(compute_epsilon_for_sample, task_arg) for task_arg in batch]
            
            # Process results from this batch
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                processed += 1
                if processed % 1000 == 0 or processed == total_samples:
                    elapsed = time.time() - start_time
                    rate = processed / elapsed if elapsed > 0 else 0
                    eta = (total_samples - processed) / rate if rate > 0 else 0
                    print(f"Progress: {processed}/{total_samples} samples ({100*processed/total_samples:.1f}%) | "
                          f"Rate: {rate:.1f} samples/s | ETA: {eta/60:.1f} min")
    
    # Sort by sample index and extract epsilons
    results.sort(key=lambda x: x[0])
    per_sample_epsilons = np.array([eps for _, eps in results])
    
    # Save results
    out_file = args.out or os.path.join(args.data_dir, 'per_sample_epsilons.npy')
    np.save(out_file, per_sample_epsilons)
    print(f"\nSaved per-sample epsilons to: {out_file}")
    
    # Print statistics
    print(f"\nStatistics:")
    print(f"  Mean epsilon: {np.mean(per_sample_epsilons):.4f}")
    print(f"  Median epsilon: {np.median(per_sample_epsilons):.4f}")
    print(f"  Max epsilon: {np.max(per_sample_epsilons):.4f}")
    print(f"  Min epsilon: {np.min(per_sample_epsilons):.4f}")
    print(f"  Std epsilon: {np.std(per_sample_epsilons):.4f}")
    
    # Find most vulnerable samples
    top_k = 10
    top_indices = np.argsort(per_sample_epsilons)[-top_k:][::-1]
    print(f"\nTop {top_k} most vulnerable samples:")
    for rank, idx in enumerate(top_indices, 1):
        print(f"  {rank}. Sample {idx}: epsilon = {per_sample_epsilons[idx]:.4f}")


if __name__ == '__main__':
    main()
