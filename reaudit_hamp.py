"""
Re-run the HAMP LOO-LR audit on already-saved binary vectors.

Usage:
    python reaudit_hamp.py <out_dir>

Loads binary_vectors_in.npy and binary_vectors_out.npy from <out_dir>,
re-runs the LOO logistic regression with an exactly-balanced training set per
fold (fixes the LOO class-imbalance artifact that inflates eps when there is no
real signal), saves new scores_in.npy / scores_out.npy, and prints the GDP
no-holdout eps.

Run print_tradeoff.py on <out_dir> afterwards to get the full
GDP/CP x no-holdout/holdout breakdown.
"""

import sys
import argparse
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).parent))
from utils.audit import compute_eps_lower_from_mia


def reaudit(out_dir, alpha=0.1, delta=1e-5):
    out_dir = Path(out_dir)
    bv_in  = np.load(out_dir / 'binary_vectors_in.npy')
    bv_out = np.load(out_dir / 'binary_vectors_out.npy')

    print(f'Loaded binary vectors: in={bv_in.shape}, out={bv_out.shape}')

    all_features = np.vstack([bv_in, bv_out])
    all_labels   = np.concatenate([
        np.ones(len(bv_in),  dtype=np.int64),
        np.zeros(len(bv_out), dtype=np.int64),
    ])

    in_idx  = np.where(all_labels == 1)[0]
    out_idx = np.where(all_labels == 0)[0]

    scores = np.zeros(len(all_features))
    for i in range(len(all_features)):
        # Build a balanced training set: remove sample i, then subsample the
        # majority class down to match the minority class size.  This ensures
        # the LR intercept is not biased by class imbalance (N-1 vs N), which
        # would create artifactually separated score clusters even for
        # uninformative (all-zero) binary vectors.
        rng = np.random.default_rng(seed=i)
        if all_labels[i] == 1:
            train_in  = in_idx[in_idx != i]           # N-1 in-world
            train_out = rng.choice(out_idx, size=len(train_in), replace=False)
        else:
            train_out = out_idx[out_idx != i]          # N-1 out-world
            train_in  = rng.choice(in_idx, size=len(train_out), replace=False)

        train_idx = np.concatenate([train_in, train_out])
        X_train = all_features[train_idx]
        y_train = all_labels[train_idx]

        clf = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs')
        clf.fit(X_train, y_train)
        scores[i] = clf.predict_proba(all_features[i:i+1])[0, 1]

    scores_in  = scores[all_labels == 1]
    scores_out = scores[all_labels == 0]
    print(f'scores_in:  n={len(scores_in)}  mean={scores_in.mean():.4f}  std={scores_in.std():.4f}')
    print(f'scores_out: n={len(scores_out)}  mean={scores_out.mean():.4f}  std={scores_out.std():.4f}')

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
