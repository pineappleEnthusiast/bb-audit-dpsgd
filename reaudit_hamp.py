"""
Re-run the HAMP audit on already-saved binary vectors.

Usage:
    python reaudit_hamp.py <out_dir>

Loads binary_vectors_in.npy and binary_vectors_out.npy from <out_dir>,
scores each model by its mean binary correctness across all 18 augmentations
(fraction of augmentations where the canary's wrong label is predicted),
saves new scores_in.npy / scores_out.npy, and prints the GDP no-holdout eps.

Run print_tradeoff.py on <out_dir> afterwards to get the full
GDP/CP x no-holdout/holdout breakdown.
"""

import sys
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils.audit import compute_eps_lower_from_mia


def reaudit(out_dir, alpha=0.1, delta=1e-5):
    out_dir = Path(out_dir)
    bv_in  = np.load(out_dir / 'binary_vectors_in.npy')
    bv_out = np.load(out_dir / 'binary_vectors_out.npy')

    print(f'Loaded binary vectors: in={bv_in.shape}, out={bv_out.shape}')

    all_labels = np.concatenate([
        np.ones(len(bv_in),  dtype=np.int64),
        np.zeros(len(bv_out), dtype=np.int64),
    ])

    scores_in  = bv_in.mean(axis=1)
    scores_out = bv_out.mean(axis=1)
    print(f'scores_in:  n={len(scores_in)}  mean={scores_in.mean():.4f}  std={scores_in.std():.4f}')
    print(f'scores_out: n={len(scores_out)}  mean={scores_out.mean():.4f}  std={scores_out.std():.4f}')

    scores = np.concatenate([scores_in, scores_out])
    if scores_in.mean() < scores_out.mean():
        scores = -scores

    max_t, emp_eps = compute_eps_lower_from_mia(scores, all_labels, alpha, delta, method='GDP')
    emp_eps = float(emp_eps) if emp_eps is not None else 0.0
    print(f'GDP no-holdout emp_eps = {emp_eps:.6f}')

    np.save(out_dir / 'scores_in.npy',  scores[all_labels == 1].astype(np.float32))
    np.save(out_dir / 'scores_out.npy', scores[all_labels == 0].astype(np.float32))
    print(f'Saved scores_in.npy / scores_out.npy to {out_dir}')
    print(f'Run: python print_tradeoff.py {out_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('out_dir')
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--delta', type=float, default=1e-5)
    args = parser.parse_args()
    reaudit(args.out_dir, args.alpha, args.delta)
