"""
Compute empirical epsilon bounds for a single experiment folder.

Given a results folder with scores_in.npy/losses_in.npy and scores_out.npy/losses_out.npy,
compute empirical epsilon via GDP and CP methods with and without holdout splits.

Holdout: 25%, 50%, 75% (threshold selected on fit set, evaluated on holdout).
No holdout: fit and eval on full data (not a valid lower bound, but useful as upper bound).
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
    """Load scores_in and scores_out from directory (supports losses_* and scores_* names)."""
    in_path = os.path.join(data_dir, 'losses_in.npy')
    out_path = os.path.join(data_dir, 'losses_out.npy')
    if not os.path.exists(in_path):
        in_path = os.path.join(data_dir, 'scores_in.npy')
    if not os.path.exists(out_path):
        out_path = os.path.join(data_dir, 'scores_out.npy')

    if not os.path.exists(in_path):
        raise FileNotFoundError(f"Could not find losses_in/scores_in in {data_dir}")
    if not os.path.exists(out_path):
        raise FileNotFoundError(f"Could not find losses_out/scores_out in {data_dir}")

    scores_in = np.load(in_path).astype(np.float64)
    scores_out = np.load(out_path).astype(np.float64)
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


def _fit_best_threshold_lib(fit_si, fit_so, alpha, delta, method, m=M):
    """Find best threshold on fit set by searching over grid."""
    thresholds = _build_threshold_grid(fit_si, fit_so, m)
    scores, labels = _make_scores_labels(fit_si, fit_so)

    best_eps = 0.0
    best_threshold = None
    for t in thresholds:
        eps = compute_eps_lower_from_mia_given_t(scores, labels, alpha, delta, t, method)
        if not np.isnan(float(eps)) and float(eps) > best_eps:
            best_eps = float(eps)
            best_threshold = float(t)

    return best_threshold, best_eps


def _eval_threshold_lib(hold_si, hold_so, t, alpha, delta, method):
    """Evaluate a fixed threshold on holdout using library method."""
    if t is None:
        return 0.0
    scores, labels = _make_scores_labels(hold_si, hold_so)
    eps = compute_eps_lower_from_mia_given_t(scores, labels, alpha, delta, t, method)
    return float(eps) if not np.isnan(float(eps)) else 0.0


def _split_scores_for_holdout(scores_in, scores_out, holdout_frac, seed):
    """Split scores into fit and holdout sets."""
    n = len(scores_in)
    if n != len(scores_out):
        raise ValueError(f'Expected equal in/out sample sizes, got {n} and {len(scores_out)}')
    if not (0.0 < float(holdout_frac) < 1.0):
        raise ValueError(f'holdout_frac must be in (0, 1), got {holdout_frac}')
    if n < 2:
        raise ValueError('Need at least 2 scores per side for fit/holdout split')

    rng = np.random.default_rng(int(seed))
    indices = rng.permutation(n)
    n_holdout = max(1, int(round(n * float(holdout_frac))))
    n_holdout = min(n_holdout, n - 1)
    hold_idx = indices[:n_holdout]
    fit_idx = indices[n_holdout:]
    return (
        scores_in[fit_idx],
        scores_out[fit_idx],
        scores_in[hold_idx],
        scores_out[hold_idx],
    )


def compute_eps_with_holdout(scores_in, scores_out, holdout_frac, method='GDP', seed=0, m=M):
    """
    Compute eps by fitting threshold on fit set, evaluating on holdout.

    method: 'GDP' or 'cp'
    """
    fit_si, fit_so, hold_si, hold_so = _split_scores_for_holdout(scores_in, scores_out, holdout_frac, seed)

    # Find best threshold on fit set
    best_threshold, _ = _fit_best_threshold_lib(fit_si, fit_so, GAMMA, DELTA, method, m=m)

    # Evaluate on holdout
    holdout_eps = _eval_threshold_lib(hold_si, hold_so, best_threshold, GAMMA, DELTA, method)
    return holdout_eps


def compute_eps_no_holdout(scores_in, scores_out, method='GDP', m=M):
    """
    Compute eps by fitting and evaluating on full data (not a valid lower bound).
    """
    # Find best threshold on full data
    best_threshold, best_eps = _fit_best_threshold_lib(scores_in, scores_out, GAMMA, DELTA, method, m=m)
    return best_eps


def _find_top_k_thresholds_on_fit(fit_si, fit_so, k, method='GDP', m=M):
    """Find top-k thresholds ranked by eps on fit set."""
    thresholds = _build_threshold_grid(fit_si, fit_so, m)
    scores, labels = _make_scores_labels(fit_si, fit_so)

    eps_by_threshold = []
    for t in thresholds:
        eps = compute_eps_lower_from_mia_given_t(scores, labels, GAMMA, DELTA, t, method)
        if not np.isnan(float(eps)):
            eps_by_threshold.append((float(eps), float(t)))

    eps_by_threshold.sort(reverse=True)
    return [t for _, t in eps_by_threshold[:k]]


def compute_eps_top_k_holdout(scores_in, scores_out, holdout_frac, method='GDP', seed=0, k=5, m=M):
    """
    Find top-k thresholds on fit set, evaluate all on holdout, report max eps.

    This tests robustness: instead of trusting a single fit-set threshold,
    try multiple and see which generalizes best to holdout.
    """
    fit_si, fit_so, hold_si, hold_so = _split_scores_for_holdout(scores_in, scores_out, holdout_frac, seed)

    # Find top-k thresholds on fit set
    top_thresholds = _find_top_k_thresholds_on_fit(fit_si, fit_so, k, method, m=m)

    # Evaluate each on holdout set
    holdout_eps_vals = []
    for t in top_thresholds:
        eps = _eval_threshold_lib(hold_si, hold_so, t, GAMMA, DELTA, method)
        holdout_eps_vals.append(eps)

    max_holdout_eps = float(np.max(holdout_eps_vals)) if holdout_eps_vals else 0.0
    return max_holdout_eps


def main():
    parser = argparse.ArgumentParser(
        description='Compute empirical epsilon bounds (GDP and CP, with/without holdout).'
    )
    parser.add_argument(
        'data_dir',
        help='Directory containing losses_in.npy/scores_in.npy and losses_out.npy/scores_out.npy',
    )
    parser.add_argument('--seed', type=int, default=0, help='Random seed for holdout split')
    parser.add_argument('--m', type=int, default=M, help='Number of threshold grid points')
    args = parser.parse_args()

    scores_in, scores_out = _load_scores(args.data_dir)

    print(f"Data dir: {args.data_dir}")
    print(f"scores_in:  n={len(scores_in)}  mean={scores_in.mean():.4f}  std={scores_in.std():.4f}")
    print(f"scores_out: n={len(scores_out)}  mean={scores_out.mean():.4f}  std={scores_out.std():.4f}")
    print(f"delta={DELTA}  gamma={GAMMA}  m={args.m}\n")

    # Sign-flip so larger scores always mean "more likely to be IN".
    si, so = scores_in.copy(), scores_out.copy()
    if si.mean() < so.mean():
        si, so = -si, -so

    results = {}

    # No holdout (upper bound)
    results['GDP no holdout'] = compute_eps_no_holdout(si, so, method='GDP', m=args.m)
    results['CP no holdout'] = compute_eps_no_holdout(si, so, method='cp', m=args.m)

    # Holdout splits
    for holdout_pct in [25, 50, 75]:
        holdout_frac = float(holdout_pct) / 100.0
        results[f'GDP {holdout_pct}%'] = compute_eps_with_holdout(si, so, holdout_frac, method='GDP', seed=args.seed, m=args.m)
        results[f'CP {holdout_pct}%'] = compute_eps_with_holdout(si, so, holdout_frac, method='cp', seed=args.seed, m=args.m)

    # Print first table (standard evaluation)
    print(f"{'Method':<20}  {'Empirical eps':>14}")
    print('-' * 37)
    for label in ['GDP no holdout', 'GDP 25%', 'GDP 50%', 'GDP 75%', 'CP no holdout', 'CP 25%', 'CP 50%', 'CP 75%']:
        eps = results[label]
        print(f"{label:<20}  {eps:14.6f}")

    # Compute second table (top-5 thresholds on fit, best on holdout)
    print(f"\n(Top-5 evaluation: top-5 thresholds from fit set, max eps on holdout)")
    top_k_results = {}
    for holdout_pct in [25, 50, 75]:
        holdout_frac = float(holdout_pct) / 100.0
        top_k_results[f'GDP {holdout_pct}%'] = compute_eps_top_k_holdout(si, so, holdout_frac, method='GDP', seed=args.seed, k=5, m=args.m)
        top_k_results[f'CP {holdout_pct}%'] = compute_eps_top_k_holdout(si, so, holdout_frac, method='cp', seed=args.seed, k=5, m=args.m)

    # Print second table
    print(f"\n{'Method':<20}  {'Empirical eps':>14}")
    print('-' * 37)
    for label in ['GDP 25%', 'GDP 50%', 'GDP 75%', 'CP 25%', 'CP 50%', 'CP 75%']:
        eps = top_k_results[label]
        print(f"{label:<20}  {eps:14.6f}")


if __name__ == '__main__':
    main()
