import sys
import os
import numpy as np
from scipy.stats import beta as beta_dist
sys.path.insert(0, '.')
from utils.audit import compute_eps_lower_single, compute_eps_lower_from_mia, compute_eps_lower_from_mia_given_t, AttackResults


def _lower_convex_envelope(alphas, betas):
    """
    Given points (alpha_i, beta_i) return the lower convex envelope as sorted
    (alpha, beta) pairs. Adds boundary points (0,1) and (1,0).
    Uses the lower-left convex hull: for each alpha, the minimum achievable beta.
    """
    pts = list(zip(alphas, betas)) + [(0.0, 1.0), (1.0, 0.0)]
    pts = sorted(set(pts))

    # Lower convex hull (monotone chain on lower side)
    hull = []
    for p in pts:
        while len(hull) >= 2 and (
            (hull[-1][0] - hull[-2][0]) * (p[1] - hull[-2][1]) <=
            (p[0] - hull[-2][0]) * (hull[-1][1] - hull[-2][1])
        ):
            hull.pop()
        hull.append(p)
    return hull


def _cp_upper(x, n, gamma):
    """One-sided Clopper-Pearson upper bound at level gamma.
    x successes out of n trials → upper bound on true proportion."""
    if n == 0:
        return 1.0
    if x == n:
        return 1.0
    return float(beta_dist.ppf(1.0 - gamma, x + 1, n - x))


def _best_threshold_from_envelope(raw_pts, envelope, delta):
    """
    Find the threshold t* and reflected flag that achieves max eps_lb on the envelope.
    raw_pts: list of (a, b, t, reflected) from fit set (original + symmetrized).
    Returns (best_t, reflected, best_eps).
    """
    best_eps, best_t, best_reflected = 0.0, None, False
    for a, b in envelope:
        if a > 0 and (1.0 - delta - b) > 0:
            eps = np.log((1.0 - delta - b) / a)
            if eps > best_eps:
                best_eps = eps
                dists = [abs(p[0] - a) + abs(p[1] - b) for p in raw_pts]
                best_i = int(np.argmin(dists))
                best_t, best_reflected = raw_pts[best_i][2], raw_pts[best_i][3]
    return best_t, best_reflected, best_eps


def _eval_on_holdout_no_cp(hold_si, hold_so, t, reflected, delta):
    if reflected:
        a = float(np.mean(hold_si <  t))   # swapped: treat in as out
        b = float(np.mean(hold_so >= t))
    else:
        a = float(np.mean(hold_so >= t))   # FPR
        b = float(np.mean(hold_si <  t))   # FNR
    if a > 0 and (1.0 - delta - b) > 0:
        return float(np.log((1.0 - delta - b) / a))
    return 0.0


def _eval_on_holdout_cp(hold_si, hold_so, t, reflected, delta, gamma):
    n_in, n_out = len(hold_si), len(hold_so)
    if reflected:
        fp = int(np.sum(hold_si <  t))
        fn = int(np.sum(hold_so >= t))
        a  = _cp_upper(fp, n_in,  gamma)
        b  = _cp_upper(fn, n_out, gamma)
    else:
        fp = int(np.sum(hold_so >= t))
        fn = int(np.sum(hold_si <  t))
        a  = _cp_upper(fp, n_out, gamma)
        b  = _cp_upper(fn, n_in,  gamma)
    if a > 0 and (1.0 - delta - b) > 0:
        return float(np.log((1.0 - delta - b) / a))
    return 0.0


def compute_eps_tradeoff_with_cp(fit_si, fit_so, hold_si, hold_so, delta, gamma):
    """
    Algorithm 5 (EmpiricalDPLB) with Clopper-Pearson correction.

    For each threshold t: compute CP upper bounds on FPR and FNR.
    Symmetrize by adding reflected points (1-beta_hat, 1-alpha_hat) and tracking
    which direction each point came from.
    Selects t* + direction on fit set, evaluates using the same direction on holdout.
    """
    thresholds = np.sort(np.unique(np.concatenate([fit_si, fit_so])))
    n_in_fit, n_out_fit = len(fit_si), len(fit_so)

    raw_pts = []
    for t in thresholds:
        fp = int(np.sum(fit_so >= t))
        fn = int(np.sum(fit_si <  t))
        a  = _cp_upper(fp, n_out_fit, gamma)
        b  = _cp_upper(fn, n_in_fit,  gamma)
        raw_pts.append((a,       b,       t, False))   # original
        raw_pts.append((1.0 - b, 1.0 - a, t, True))   # symmetrized

    alphas = np.array([p[0] for p in raw_pts])
    betas  = np.array([p[1] for p in raw_pts])
    envelope = _lower_convex_envelope(alphas, betas)

    best_t, reflected, _ = _best_threshold_from_envelope(raw_pts, envelope, delta)
    if best_t is None:
        return 0.0
    return _eval_on_holdout_cp(hold_si, hold_so, best_t, reflected, delta, gamma)


def compute_eps_tradeoff_no_cp(fit_si, fit_so, hold_si, hold_so, delta):
    """
    Algorithm 5 (EmpiricalDPLB) without Clopper-Pearson correction, with holdout.

    Fit set: select threshold t* = argmax eps_lb over raw (FPR, FNR) pairs.
    Holdout set: evaluate at fixed t* using raw frequencies (no CP correction).

    Symmetry enforced by adding reflected points (1-beta_i, 1-alpha_i).
    Convex lower envelope computed over all points.

    eps_lb(delta) = max_{(alpha, beta) on convex envelope, alpha > 0, 1-delta-beta > 0}
                    log((1 - delta - beta) / alpha)
    """
    thresholds = np.sort(np.unique(np.concatenate([fit_si, fit_so])))

    # Step 1: select t* on fit set (track reflected flag per point)
    raw_pts = []
    for t in thresholds:
        a = float(np.mean(fit_so >= t))   # FPR
        b = float(np.mean(fit_si <  t))   # FNR
        raw_pts.append((a,       b,       t, False))   # original
        raw_pts.append((1.0 - b, 1.0 - a, t, True))   # symmetrized

    alphas = np.array([p[0] for p in raw_pts])
    betas  = np.array([p[1] for p in raw_pts])
    envelope_fit = _lower_convex_envelope(alphas, betas)

    best_t, reflected, _ = _best_threshold_from_envelope(raw_pts, envelope_fit, delta)
    if best_t is None:
        return 0.0

    # Step 2: evaluate at fixed t* on holdout using same direction
    return _eval_on_holdout_no_cp(hold_si, hold_so, best_t, reflected, delta)


# ---------------------------------------------------------------------------

d        = sys.argv[1] if len(sys.argv) > 1 else 'tradeoff_curves/mnist_no_defense_eps10/mnist_cnn_eps10.0'
alpha    = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
delta    = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-5
holdout  = '--no-holdout' not in sys.argv

holdout_frac = 0.5
for arg in sys.argv:
    if arg.startswith('--holdout-frac='):
        holdout_frac = float(arg.split('=')[1])

scores_in  = np.load(f'{d}/losses_in.npy')  if os.path.exists(f'{d}/losses_in.npy')  else np.load(f'{d}/scores_in.npy')
scores_out = np.load(f'{d}/losses_out.npy') if os.path.exists(f'{d}/losses_out.npy') else np.load(f'{d}/scores_out.npy')

print(f"scores_in:  n={len(scores_in)}  mean={scores_in.mean():.4f}  std={scores_in.std():.4f}")
print(f"scores_out: n={len(scores_out)}  mean={scores_out.mean():.4f}  std={scores_out.std():.4f}")
print(f"holdout={holdout}  holdout_frac={holdout_frac}")

# sign-flip so higher score = member
si, so = scores_in.copy(), scores_out.copy()
if si.mean() < so.mean():
    si, so = -si, -so

n = len(si)
np.random.seed(0)
idx = np.random.permutation(n)

if not holdout:
    fit_idx = hold_idx = idx
elif holdout_frac == 0.5:
    fit_idx, hold_idx = idx[:n//2], idx[n//2:]
else:
    n_hold = max(1, int(n * holdout_frac))
    fit_idx, hold_idx = idx[n_hold:], idx[:n_hold]

fit_scores  = np.concatenate([si[fit_idx],  so[fit_idx]]).astype(np.float32)
fit_labels  = np.concatenate([np.ones(len(fit_idx)), np.zeros(len(fit_idx))]).astype(np.int64)
hold_scores = np.concatenate([si[hold_idx], so[hold_idx]]).astype(np.float32)
hold_labels = np.concatenate([np.ones(len(hold_idx)), np.zeros(len(hold_idx))]).astype(np.int64)

print(f"  fit={len(fit_idx)} reps  hold={len(hold_idx)} reps\n")

# GDP and Clopper-Pearson (with holdout)
for method, label in [('GDP', 'GDP'), ('cp', 'Clopper-Pearson')]:
    t, _ = compute_eps_lower_from_mia(fit_scores, fit_labels, alpha, delta, method, n_procs=4)
    emp_eps = compute_eps_lower_from_mia_given_t(hold_scores, hold_labels, alpha, delta, t, method)
    print(f"{label:30s}:  emp_eps={emp_eps:.4f}  threshold={t:.4f}")

# Algorithm 5 without CP correction
eps_no_cp = compute_eps_tradeoff_no_cp(si[fit_idx], so[fit_idx],
                                        si[hold_idx], so[hold_idx], delta)
print(f"{'Alg5 (no CP correction)':30s}:  emp_eps={eps_no_cp:.4f}")

# Algorithm 5 with CP correction
eps_with_cp = compute_eps_tradeoff_with_cp(si[fit_idx], so[fit_idx],
                                            si[hold_idx], so[hold_idx], delta, gamma=alpha)
print(f"{'Alg5 (with CP correction)':30s}:  emp_eps={eps_with_cp:.4f}")

# Per-decile table
print(f"\nPer-decile (Clopper-Pearson):")
print(f"{'t':>10}  {'TP':>6}  {'FP':>6}  {'TN':>6}  {'FN':>6}  {'TPR':>6}  {'FPR':>6}  {'eps_lb':>8}")
print('-' * 70)
all_scores = np.concatenate([si, so]).astype(np.float32)
all_labels = np.concatenate([np.ones(len(si)), np.zeros(len(so))]).astype(np.int64)
threshs = np.percentile(all_scores, np.arange(0, 101, 10))
n_in, n_out = len(si), len(so)
for t in threshs:
    tp = int(np.sum(all_scores[all_labels == 1] >= t))
    fp = int(np.sum(all_scores[all_labels == 0] >= t))
    fn = n_in - tp
    tn = n_out - fp
    r  = AttackResults(FN=fn, FP=fp, TN=tn, TP=tp)
    eps = compute_eps_lower_single(r, alpha, delta, 'cp')
    print(f"{t:10.4f}  {tp:6d}  {fp:6d}  {tn:6d}  {fn:6d}  {tp/n_in:6.3f}  {fp/n_out:6.3f}  {eps:8.4f}")
