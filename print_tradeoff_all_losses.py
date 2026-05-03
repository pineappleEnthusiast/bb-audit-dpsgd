"""
Compute per-sample empirical epsilon using all_losses_in/out.npy.

Given a results folder with all_losses_in.npy and all_losses_out.npy of shape
(n_reps, n_samples), compute per-sample empirical epsilon via GDP or CP and
report the maximum (worst-case exposed sample).

No holdout: fit and eval on full data (upper bound).
Holdout 50%: threshold fit on 50% of reps, evaluated on remaining 50%.
"""

import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.audit import compute_eps_lower_from_mia, compute_eps_lower_from_mia_given_t

DELTA = 1e-5
GAMMA = 0.1
M = 200  # number of threshold grid points


def _load_scores(data_dir):
    """Load all_losses_in and all_losses_out, return as float64 arrays of shape (n_reps, n_samples)."""
    in_path = os.path.join(data_dir, 'all_losses_in.npy')
    out_path = os.path.join(data_dir, 'all_losses_out.npy')

    if not os.path.exists(in_path):
        raise FileNotFoundError(f"Could not find all_losses_in.npy in {data_dir}")
    if not os.path.exists(out_path):
        raise FileNotFoundError(f"Could not find all_losses_out.npy in {data_dir}")

    scores_in = np.load(in_path, allow_pickle=True).astype(np.float64)
    scores_out = np.load(out_path, allow_pickle=True).astype(np.float64)
    return scores_in, scores_out


def _build_threshold_grid(scores_in, scores_out, m):
    """Build threshold grid as percentiles across both in and out."""
    all_scores = np.concatenate([scores_in, scores_out])
    percentiles = np.linspace(0.0, 100.0, num=int(m), endpoint=True)
    return np.percentile(all_scores, percentiles)


def _make_scores_labels(scores_in, scores_out):
    """Concatenate in/out scores and labels (1 for in, 0 for out)."""
    scores = np.concatenate([scores_in, scores_out]).astype(np.float32)
    labels = np.concatenate([np.ones(len(scores_in)), np.zeros(len(scores_out))]).astype(np.int64)
    return scores, labels


def _compute_eps_no_holdout_1d(si, so, method='GDP'):
    """Compute no-holdout epsilon for a single sample's in/out score vectors."""
    thresholds = _build_threshold_grid(si, so, M)
    scores, labels = _make_scores_labels(si, so)
    best_eps = 0.0
    for t in thresholds:
        eps = compute_eps_lower_from_mia_given_t(scores, labels, GAMMA, DELTA, t, method)
        v = float(eps)
        if not np.isnan(v) and v > best_eps:
            best_eps = v
    return best_eps


def _compute_eps_holdout_1d(si, so, holdout_frac=0.5, seed=0, method='GDP'):
    """Compute holdout epsilon for a single sample's in/out score vectors."""
    n = len(si)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_hold = max(1, min(int(round(n * holdout_frac)), n - 1))
    hold_idx, fit_idx = idx[:n_hold], idx[n_hold:]

    fit_si, fit_so = si[fit_idx], so[fit_idx]
    hold_si, hold_so = si[hold_idx], so[hold_idx]

    thresholds = _build_threshold_grid(fit_si, fit_so, M)
    fit_scores, fit_labels = _make_scores_labels(fit_si, fit_so)
    best_t, best_eps_fit = None, 0.0
    for t in thresholds:
        eps = compute_eps_lower_from_mia_given_t(fit_scores, fit_labels, GAMMA, DELTA, t, method)
        v = float(eps)
        if not np.isnan(v) and v > best_eps_fit:
            best_eps_fit, best_t = v, t

    if best_t is None:
        return 0.0

    hold_scores, hold_labels = _make_scores_labels(hold_si, hold_so)
    eps = compute_eps_lower_from_mia_given_t(hold_scores, hold_labels, GAMMA, DELTA, best_t, method)
    v = float(eps)
    return v if not np.isnan(v) else 0.0


def main():
    parser = argparse.ArgumentParser(
        description='Compute per-sample empirical epsilon using all_losses_in/out.npy and report the max.'
    )
    parser.add_argument(
        'data_dir',
        help='Directory containing all_losses_in.npy and all_losses_out.npy of shape (n_reps, n_samples)',
    )
    parser.add_argument('--seed', type=int, default=0, help='Random seed for holdout split')
    parser.add_argument('--method', type=str, default='GDP', choices=['GDP', 'cp'],
                        help='Epsilon computation method')
    args = parser.parse_args()

    all_in, all_out = _load_scores(args.data_dir)
    n_reps, n_samples = all_in.shape

    print(f"Data dir: {args.data_dir}")
    print(f"Shape: ({n_reps} reps, {n_samples} samples)  delta={DELTA}  gamma={GAMMA}")
    print(f"Method: {args.method}")
    print(f"Computing per-sample epsilon (no-holdout and 50% holdout)...\n")

    # Sign-flip convention: ensure in-world has higher loss on average
    if all_in.mean() < all_out.mean():
        all_in, all_out = -all_in, -all_out

    no_holdout_eps = np.zeros(n_samples)
    holdout_eps    = np.zeros(n_samples)

    for i in range(n_samples):
        si, so = all_in[:, i], all_out[:, i]
        no_holdout_eps[i] = _compute_eps_no_holdout_1d(si, so, method=args.method)
        holdout_eps[i]    = _compute_eps_holdout_1d(si, so, holdout_frac=0.5, seed=args.seed, method=args.method)

        if (i + 1) % 5000 == 0:
            print(f"  [{i+1}/{n_samples}] max no-holdout so far: {no_holdout_eps[:i+1].max():.4f}  "
                  f"max holdout so far: {holdout_eps[:i+1].max():.4f}")

    max_nh_idx = int(np.argmax(no_holdout_eps))
    max_h_idx  = int(np.argmax(holdout_eps))

    print(f"\n{'='*55}")
    print(f"{'Metric':<32}  {'Value':>10}  {'Sample idx':>10}")
    print('-' * 55)
    print(f"{'Max eps (no holdout)':<32}  {no_holdout_eps[max_nh_idx]:10.6f}  {max_nh_idx:>10}")
    print(f"{'Max eps (50% holdout)':<32}  {holdout_eps[max_h_idx]:10.6f}  {max_h_idx:>10}")
    print(f"{'Mean eps (no holdout)':<32}  {no_holdout_eps.mean():10.6f}")
    print(f"{'Mean eps (50% holdout)':<32}  {holdout_eps.mean():10.6f}")
    print(f"{'Median eps (no holdout)':<32}  {np.median(no_holdout_eps):10.6f}")
    print(f"{'% samples with eps>0 (no holdout)':<32}  {(no_holdout_eps > 0).mean()*100:9.2f}%")


if __name__ == '__main__':
    main()
