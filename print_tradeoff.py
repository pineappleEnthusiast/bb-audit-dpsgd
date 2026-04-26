import sys
import os
import numpy as np
sys.path.insert(0, '.')
from utils.audit import compute_eps_lower_single, compute_eps_lower_from_mia, compute_eps_lower_from_mia_given_t, AttackResults
from parallel_audit_multi_canary import _audit_from_scores

d        = sys.argv[1] if len(sys.argv) > 1 else 'tradeoff_curves/mnist_no_defense_eps10/mnist_cnn_eps10.0'
alpha    = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
delta    = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-5
holdout  = '--no-holdout' not in sys.argv

# --holdout-frac 0.25 uses 75% to fit threshold, 25% to evaluate (default 0.5)
holdout_frac = 0.5
for arg in sys.argv:
    if arg.startswith('--holdout-frac='):
        holdout_frac = float(arg.split('=')[1])

scores_in  = np.load(f'{d}/losses_in.npy')  if os.path.exists(f'{d}/losses_in.npy')  else np.load(f'{d}/scores_in.npy')
scores_out = np.load(f'{d}/losses_out.npy') if os.path.exists(f'{d}/losses_out.npy') else np.load(f'{d}/scores_out.npy')

print(f"scores_in:  n={len(scores_in)}  mean={scores_in.mean():.4f}  std={scores_in.std():.4f}")
print(f"scores_out: n={len(scores_out)}  mean={scores_out.mean():.4f}  std={scores_out.std():.4f}")
print(f"holdout={holdout}  holdout_frac={holdout_frac}")

if not holdout or holdout_frac == 0.5:
    emp_eps, t, _, _ = _audit_from_scores(scores_in, scores_out, alpha=alpha, delta=delta,
                                           holdout_audit=holdout, seed=0)
else:
    # Custom holdout fraction: fit on (1-frac), evaluate on frac
    n = len(scores_in)
    np.random.seed(0)
    indices = np.random.permutation(n)
    n_holdout = max(1, int(n * holdout_frac))
    fit_idx  = indices[n_holdout:]
    hold_idx = indices[:n_holdout]

    # Flip sign if needed
    if scores_in.mean() < scores_out.mean():
        scores_in, scores_out = -scores_in, -scores_out

    fit_scores  = np.concatenate([scores_in[fit_idx],  scores_out[fit_idx]])
    fit_labels  = np.concatenate([np.ones(len(fit_idx)), np.zeros(len(fit_idx))]).astype(np.int64)
    t, _        = compute_eps_lower_from_mia(fit_scores, fit_labels, alpha, delta, 'GDP', n_procs=1)

    hold_scores = np.concatenate([scores_in[hold_idx], scores_out[hold_idx]])
    hold_labels = np.concatenate([np.ones(len(hold_idx)), np.zeros(len(hold_idx))]).astype(np.int64)
    emp_eps     = compute_eps_lower_from_mia_given_t(hold_scores, hold_labels, alpha, delta, t, 'GDP')
    print(f"  fit on {len(fit_idx)} reps, evaluate on {n_holdout} reps")

print(f"\nemp_eps={emp_eps:.4f}  threshold={t:.4f}")

# Per-decile TP/FP/TN/FN table
si, so = scores_in.copy(), scores_out.copy()
if si.mean() < so.mean():
    si, so = -si, -so
scores = np.concatenate([si, so]).astype(np.float32)
labels = np.concatenate([np.ones(len(si)), np.zeros(len(so))]).astype(np.int64)
threshs = np.percentile(scores, np.arange(0, 101, 10))
n_in, n_out = len(si), len(so)

print(f"\n{'t':>10}  {'TP':>6}  {'FP':>6}  {'TN':>6}  {'FN':>6}  {'TPR':>6}  {'FPR':>6}  {'eps_lb':>8}")
print('-' * 70)
for t in threshs:
    tp = int(np.sum(scores[labels == 1] >= t))
    fp = int(np.sum(scores[labels == 0] >= t))
    fn = n_in - tp
    tn = n_out - fp
    r  = AttackResults(FN=fn, FP=fp, TN=tn, TP=tp)
    eps = compute_eps_lower_single(r, alpha, delta, 'GDP')
    print(f"{t:10.4f}  {tp:6d}  {fp:6d}  {tn:6d}  {fn:6d}  {tp/n_in:6.3f}  {fp/n_out:6.3f}  {eps:8.4f}")
