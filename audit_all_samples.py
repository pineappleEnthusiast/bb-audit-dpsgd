import numpy as np
import os
import argparse
from multiprocessing import Pool, cpu_count
from utils.audit import compute_eps_lower_from_mia

def compute_epsilon_for_sample(args):
    """Compute empirical epsilon for a single sample."""
    sample_idx, losses_in_col, losses_out_col, alpha, delta = args
    
    # Negate losses (higher loss = more likely to be member)
    scores_in = -losses_in_col
    scores_out = -losses_out_col
    
    # Combine scores and labels
    mia_scores = np.concatenate([scores_in, scores_out])
    mia_labels = np.concatenate([
        np.ones(len(scores_in)),
        np.zeros(len(scores_out))
    ])
    
    # Compute empirical epsilon
    _, emp_eps = compute_eps_lower_from_mia(
        mia_scores[mia_labels == 1],  # in-distribution scores
        mia_scores[mia_labels == 0],  # out-of-distribution scores
        alpha,
        delta,
        use_holdout=False,
        seed=0
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
    
    # Prepare arguments for parallel processing
    print(f"\nComputing empirical epsilon for {n_samples} samples...")
    task_args = [
        (i, all_losses_in[:, i], all_losses_out[:, i], args.alpha, args.delta)
        for i in range(n_samples)
    ]
    
    # Parallel computation
    n_procs = args.n_procs or cpu_count()
    print(f"Using {n_procs} processes")
    
    with Pool(n_procs) as pool:
        results = pool.map(compute_epsilon_for_sample, task_args)
    
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
