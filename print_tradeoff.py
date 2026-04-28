import argparse
import os
import numpy as np
from scipy.stats import beta as beta_dist, norm
from scipy.optimize import root_scalar


def _lower_convex_envelope(alphas, betas):
    """
    Return the lower convex envelope of the provided (alpha, beta) points.

    Boundary points (0, 1) and (1, 0) are added so epsilon search can safely
    traverse the full tradeoff frontier.
    """
    pts = list(zip(alphas, betas)) + [(0.0, 1.0), (1.0, 0.0)]
    pts = sorted(set((float(a), float(b)) for a, b in pts))

    hull = []
    for p in pts:
        while len(hull) >= 2 and (
            (hull[-1][0] - hull[-2][0]) * (p[1] - hull[-2][1]) <=
            (p[0] - hull[-2][0]) * (hull[-1][1] - hull[-2][1])
        ):
            hull.pop()
        hull.append(p)
    return hull


def _symmetrize_and_dedup(points):
    """
    Add reflected error-pair points (beta, alpha) and deduplicate by alpha,
    keeping the smallest beta for each alpha.
    """
    best_beta_by_alpha = {}

    for alpha, beta, threshold, fp, fn in points:
        candidates = [
            (float(alpha), float(beta), float(threshold), fp, fn),
            (float(beta), float(alpha), float(threshold), fp, fn),
        ]
        for a, b, t, fp_val, fn_val in candidates:
            key = round(float(a), 12)
            prev = best_beta_by_alpha.get(key)
            if prev is None or b < prev[0]:
                best_beta_by_alpha[key] = (float(b), float(t), fp_val, fn_val)

    deduped = []
    for key in sorted(best_beta_by_alpha.keys()):
        beta, threshold, fp, fn = best_beta_by_alpha[key]
        deduped.append((float(key), float(beta), float(threshold), fp, fn))
    return deduped


def _dedup_by_alpha(points):
    """Deduplicate points by alpha, keeping the smallest beta for each alpha."""
    best_beta_by_alpha = {}
    for alpha, beta, threshold, fp, fn in points:
        key = round(float(alpha), 12)
        prev = best_beta_by_alpha.get(key)
        if prev is None or beta < prev[0]:
            best_beta_by_alpha[key] = (float(beta), float(threshold), fp, fn)

    deduped = []
    for key in sorted(best_beta_by_alpha.keys()):
        beta, threshold, fp, fn = best_beta_by_alpha[key]
        deduped.append((float(key), float(beta), float(threshold), fp, fn))
    return deduped


def _symmetrize_only(points):
    """Add reflected error-pair points (beta, alpha) without hull or alpha-dedup."""
    sym_points = []
    for alpha, beta, threshold, fp, fn in points:
        sym_points.append((float(alpha), float(beta), float(threshold), fp, fn))
        sym_points.append((float(beta), float(alpha), float(threshold), fp, fn))
    return sym_points


def _cp_upper(x, n, failure_prob):
    """One-sided Clopper-Pearson upper bound with per-interval failure_prob."""
    if n == 0 or x == n:
        return 1.0
    return float(beta_dist.ppf(1.0 - failure_prob, x + 1, n - x))


def _eps_from_ab(alpha, beta, delta):
    if alpha > 0.0 and (1.0 - delta - beta) > 0.0:
        return max(0.0, float(np.log((1.0 - delta - beta) / alpha)))
    return 0.0


def _gdp_eps(fpr_ub, fnr_ub, delta):
    """Convert CP upper bounds on FPR/FNR to (eps, delta)-DP lower bound via mu-GDP.

    mu lower bound: mu_l = Phi^{-1}(1 - fpr_ub) - Phi^{-1}(fnr_ub)
    Then invert eq. (6) from the tight auditing paper (Steinke et al. 2023):
      delta = Phi(-eps/mu + mu/2) - exp(eps) * Phi(-eps/mu - mu/2)
    """
    mu = norm.ppf(1.0 - fpr_ub) - norm.ppf(fnr_ub)
    if mu <= 0.0:
        return 0.0
    def eq6(epsilon):
        return norm.cdf(-epsilon / mu + mu / 2) - np.exp(epsilon) * norm.cdf(-epsilon / mu - mu / 2) - delta
    try:
        sol = root_scalar(eq6, bracket=[0.0, 50.0], method='brentq')
        return float(sol.root)
    except Exception:
        return 0.0


def compute_gdp_eps_from_grid(scores_in, scores_out, delta, gamma, m):
    """Select best threshold on fit set using the GDP formula, return fit result dict."""
    thresholds = _build_threshold_grid(scores_in, scores_out, m)
    n_in, n_out = len(scores_in), len(scores_out)

    best_eps = 0.0
    best_threshold = None
    best_alpha = None
    best_beta = None

    for t in thresholds:
        _, _, fp, fn = _compute_alpha_beta(scores_in, scores_out, t)
        fpr_ub = _cp_upper(fp, n_out, float(gamma))
        fnr_ub = _cp_upper(fn, n_in, float(gamma))
        for a, b in [(fpr_ub, fnr_ub), (fnr_ub, fpr_ub)]:
            eps = _gdp_eps(a, b, delta)
            if eps > best_eps:
                best_eps = eps
                best_threshold = float(t)
                best_alpha = float(a)
                best_beta = float(b)

    return {'eps': best_eps, 'threshold': best_threshold, 'alpha': best_alpha, 'beta': best_beta}


def _evaluate_threshold_gdp(scores_in, scores_out, threshold, delta, gamma):
    """Evaluate a fixed threshold on holdout using the GDP formula."""
    if threshold is None:
        return {'threshold': None, 'eps': 0.0, 'alpha': None, 'beta': None, 'winner': 'n/a'}

    _, _, fp, fn = _compute_alpha_beta(scores_in, scores_out, threshold)
    n_in, n_out = len(scores_in), len(scores_out)
    fpr_ub = _cp_upper(fp, n_out, float(gamma))
    fnr_ub = _cp_upper(fn, n_in, float(gamma))

    eps_orig = _gdp_eps(fpr_ub, fnr_ub, delta)
    eps_sym = _gdp_eps(fnr_ub, fpr_ub, delta)

    if eps_sym > eps_orig:
        return {'threshold': float(threshold), 'eps': eps_sym, 'alpha': fnr_ub, 'beta': fpr_ub, 'winner': 'sym'}
    return {'threshold': float(threshold), 'eps': eps_orig, 'alpha': fpr_ub, 'beta': fnr_ub, 'winner': 'orig'}


def _split_scores_for_holdout(scores_in, scores_out, holdout_frac, seed):
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


def _kfold_splits(scores_in, scores_out, n_folds, seed):
    n = len(scores_in)
    if n != len(scores_out):
        raise ValueError(f'Expected equal in/out sample sizes, got {n} and {len(scores_out)}')
    if n_folds < 2 or n_folds > n:
        raise ValueError(f'n_folds must be in [2, n], got {n_folds}')

    rng = np.random.default_rng(int(seed))
    indices = rng.permutation(n)
    fold_boundaries = np.array_split(indices, n_folds)

    splits = []
    for i in range(n_folds):
        hold_idx = fold_boundaries[i]
        fit_idx = np.concatenate([fold_boundaries[j] for j in range(n_folds) if j != i])
        splits.append((
            scores_in[fit_idx], scores_out[fit_idx],
            scores_in[hold_idx], scores_out[hold_idx],
        ))
    return splits


def _build_threshold_grid(scores_in, scores_out, m):
    all_scores = np.concatenate([scores_in, scores_out]).astype(np.float64)
    percentiles = np.linspace(0.0, 100.0, num=int(m), endpoint=True)
    return np.percentile(all_scores, percentiles)


def _gaussian_lr_scores(scores, mu_in, sigma_in, mu_out, sigma_out):
    """Transform raw scores to log-likelihood-ratio log p(x|in) - log p(x|out) under Gaussian fit."""
    scores = np.asarray(scores, dtype=np.float64)
    log_p_in = -0.5 * ((scores - mu_in) / sigma_in) ** 2 - np.log(sigma_in)
    log_p_out = -0.5 * ((scores - mu_out) / sigma_out) ** 2 - np.log(sigma_out)
    return log_p_in - log_p_out


def apply_gaussian_lr_transform(fit_si, fit_so, hold_si, hold_so):
    """Fit Gaussians on fit set, transform all four arrays to LR scores."""
    mu_in = fit_si.mean()
    sigma_in = max(fit_si.std(), 1e-8)
    mu_out = fit_so.mean()
    sigma_out = max(fit_so.std(), 1e-8)
    return (
        _gaussian_lr_scores(fit_si, mu_in, sigma_in, mu_out, sigma_out),
        _gaussian_lr_scores(fit_so, mu_in, sigma_in, mu_out, sigma_out),
        _gaussian_lr_scores(hold_si, mu_in, sigma_in, mu_out, sigma_out),
        _gaussian_lr_scores(hold_so, mu_in, sigma_in, mu_out, sigma_out),
    )


def _compute_alpha_beta(scores_in, scores_out, threshold):
    n_in = len(scores_in)
    n_out = len(scores_out)

    fp = int(np.sum(scores_out >= threshold))
    fn = int(np.sum(scores_in < threshold))

    alpha = float(fp / n_out) if n_out > 0 else 0.0
    beta = float(fn / n_in) if n_in > 0 else 0.0
    return alpha, beta, fp, fn


def _compute_grid_points(scores_in, scores_out, thresholds, gamma, apply_cp):
    n_in = len(scores_in)
    n_out = len(scores_out)
    cp_failure_prob = float(gamma) if apply_cp else None

    pts = []
    for threshold in thresholds:
        alpha, beta, fp, fn = _compute_alpha_beta(scores_in, scores_out, threshold)
        if apply_cp:
            alpha = _cp_upper(fp, n_out, cp_failure_prob)
            beta = _cp_upper(fn, n_in, cp_failure_prob)
        pts.append((float(alpha), float(beta), float(threshold), fp, fn))
    return pts


def _best_point_from_envelope(raw_pts, envelope, delta):
    best_eps = 0.0
    best_threshold = None
    best_alpha = None
    best_beta = None

    for alpha, beta in envelope:
        eps = _eps_from_ab(alpha, beta, delta)
        if eps > best_eps:
            best_eps = eps
            best_alpha = float(alpha)
            best_beta = float(beta)

            dists = [abs(p[0] - alpha) + abs(p[1] - beta) for p in raw_pts]
            best_idx = int(np.argmin(dists))
            best_threshold = float(raw_pts[best_idx][2])

    return best_eps, best_threshold, best_alpha, best_beta


def _best_pointwise_eps(points, delta):
    best_eps = 0.0
    best_threshold = None
    best_alpha = None
    best_beta = None

    for alpha, beta, threshold, fp, fn in points:
        eps = _eps_from_ab(alpha, beta, delta)
        if eps > best_eps:
            best_eps = eps
            best_threshold = float(threshold)
            best_alpha = float(alpha)
            best_beta = float(beta)

    return {
        'eps': float(best_eps),
        'threshold': best_threshold,
        'alpha': best_alpha,
        'beta': best_beta,
        'points': points,
    }


def _evaluate_threshold(scores_in, scores_out, threshold, delta, gamma, apply_cp):
    if threshold is None:
        return {
            'threshold': None,
            'raw_alpha': None,
            'raw_beta': None,
            'cp_alpha': None,
            'cp_beta': None,
            'eps_orig': 0.0,
            'eps_sym': 0.0,
            'eps': 0.0,
            'alpha': None,
            'beta': None,
            'winner': 'n/a',
            'fp': None,
            'fn': None,
        }

    alpha_raw, beta_raw, fp, fn = _compute_alpha_beta(scores_in, scores_out, threshold)
    alpha_eval = float(alpha_raw)
    beta_eval = float(beta_raw)
    if apply_cp:
        alpha_eval = _cp_upper(fp, len(scores_out), float(gamma))
        beta_eval = _cp_upper(fn, len(scores_in), float(gamma))

    eps_orig = _eps_from_ab(alpha_eval, beta_eval, delta)
    alpha_sym = float(beta_eval)
    beta_sym = float(alpha_eval)
    eps_sym = _eps_from_ab(alpha_sym, beta_sym, delta)

    if eps_sym > eps_orig:
        chosen_alpha = alpha_sym
        chosen_beta = beta_sym
        chosen_eps = eps_sym
        chosen_side = 'sym'
    else:
        chosen_alpha = alpha_eval
        chosen_beta = beta_eval
        chosen_eps = eps_orig
        chosen_side = 'orig'

    return {
        'threshold': float(threshold),
        'raw_alpha': float(alpha_raw),
        'raw_beta': float(beta_raw),
        'cp_alpha': float(alpha_eval),
        'cp_beta': float(beta_eval),
        'eps_orig': float(eps_orig),
        'eps_sym': float(eps_sym),
        'eps': float(chosen_eps),
        'alpha': float(chosen_alpha),
        'beta': float(chosen_beta),
        'winner': chosen_side,
        'fp': int(fp),
        'fn': int(fn),
    }


def compute_eps_from_grid(scores_in, scores_out, delta, gamma, m, apply_cp, symmetrize_fit=True):
    thresholds = _build_threshold_grid(scores_in, scores_out, m)
    grid_pts = _compute_grid_points(scores_in, scores_out, thresholds, gamma, apply_cp)
    raw_pts = _symmetrize_and_dedup(grid_pts) if symmetrize_fit else _dedup_by_alpha(grid_pts)
    envelope = _lower_convex_envelope(
        np.array([p[0] for p in raw_pts], dtype=np.float64),
        np.array([p[1] for p in raw_pts], dtype=np.float64),
    )
    best_eps, best_threshold, best_alpha, best_beta = _best_point_from_envelope(raw_pts, envelope, delta)

    return {
        'eps': float(best_eps),
        'threshold': best_threshold,
        'alpha': best_alpha,
        'beta': best_beta,
        'grid_points': grid_pts,
        'raw_points': raw_pts,
        'envelope': envelope,
        'thresholds': thresholds,
        'apply_cp': bool(apply_cp),
        'symmetrize_fit': bool(symmetrize_fit),
    }


def compute_pointwise_eps_from_grid(scores_in, scores_out, delta, gamma, m, apply_cp, symmetrize=False):
    thresholds = _build_threshold_grid(scores_in, scores_out, m)
    grid_pts = _compute_grid_points(scores_in, scores_out, thresholds, gamma, apply_cp)
    points = _symmetrize_only(grid_pts) if symmetrize else grid_pts
    result = _best_pointwise_eps(points, delta)
    result.update({
        'grid_points': grid_pts,
        'symmetrized': bool(symmetrize),
        'apply_cp': bool(apply_cp),
    })
    return result


def _select_fit_thresholds(fit_si, fit_so, delta, gamma, m):
    pointwise_cp = compute_pointwise_eps_from_grid(fit_si, fit_so, delta, gamma, m, apply_cp=True, symmetrize=False)
    pointwise_cp_sym = compute_pointwise_eps_from_grid(fit_si, fit_so, delta, gamma, m, apply_cp=True, symmetrize=True)
    no_cp = compute_eps_from_grid(fit_si, fit_so, delta, gamma, m, apply_cp=False, symmetrize_fit=True)
    with_cp = compute_eps_from_grid(fit_si, fit_so, delta, gamma, m, apply_cp=True, symmetrize_fit=True)
    gdp = compute_gdp_eps_from_grid(fit_si, fit_so, delta, gamma, m)
    return {
        'Pointwise CP': pointwise_cp,
        'Pointwise CP + symmetry': pointwise_cp_sym,
        'Hull (no CP) [Alg5 no CP]': no_cp,
        'Hull (with CP) [CP/Alg5]': with_cp,
        'GDP': gdp,
    }


def _load_scores(data_dir):
    in_path = os.path.join(data_dir, 'losses_in.npy')
    out_path = os.path.join(data_dir, 'losses_out.npy')
    if not os.path.exists(in_path):
        in_path = os.path.join(data_dir, 'scores_in.npy')
    if not os.path.exists(out_path):
        out_path = os.path.join(data_dir, 'scores_out.npy')

    scores_in = np.load(in_path)
    scores_out = np.load(out_path)
    return scores_in.astype(np.float64), scores_out.astype(np.float64)


def _print_result(label, result):
    threshold = f"{result['threshold']:.4f}" if result['threshold'] is not None else "n/a"
    alpha = result['alpha'] if result['alpha'] is not None else float('nan')
    beta = result['beta'] if result['beta'] is not None else float('nan')
    print(f"{label:30s}  {result['eps']:8.4f}  {threshold:>10}  {alpha:10.4f}  {beta:10.4f}")


def main():
    parser = argparse.ArgumentParser(
        description='Unified epsilon lower-bound comparison on a fixed threshold grid.'
    )
    parser.add_argument(
        'data_dir',
        nargs='?',
        default='tradeoff_curves/mnist_no_defense_eps10/mnist_cnn_eps10.0',
        help='Directory containing losses_in.npy/losses_out.npy or scores_in.npy/scores_out.npy',
    )
    parser.add_argument('--gamma', type=float, default=0.05, help='Global CI failure probability gamma')
    parser.add_argument('--delta', type=float, default=1e-5, help='Target delta')
    parser.add_argument('--m', type=int, default=200, help='Number of threshold grid points')
    parser.add_argument(
        '--holdout-fracs', type=float, nargs='+', default=[0.1, 0.25, 0.5],
        help='One or more holdout fractions to sweep (default: 0.1 0.25 0.5)',
    )
    parser.add_argument('--seed', type=int, default=0, help='Random seed for fit/holdout split')
    parser.add_argument('--use-lr', action='store_true',
                        help='Transform scores to Gaussian log-likelihood-ratio before threshold sweep')
    parser.add_argument('--all-combos', action='store_true',
                        help='Run all (raw vs LR) x holdout_frac combinations and show a pivot table')
    parser.add_argument('--n-folds', type=int, default=1,
                        help='K-fold CV: select threshold on k-1 folds, evaluate on kth with gamma/k (Bonferroni). '
                             'Takes max eps over folds. Default 1 = disabled.')
    parser.add_argument('--print-grid', action='store_true', help='Print the per-threshold alpha/beta grid')
    args = parser.parse_args()

    if int(args.m) <= 0:
        raise ValueError(f'--m must be > 0, got {args.m}')

    scores_in, scores_out = _load_scores(args.data_dir)

    print(f"scores_in:  n={len(scores_in)}  mean={scores_in.mean():.4f}  std={scores_in.std():.4f}")
    print(f"scores_out: n={len(scores_out)}  mean={scores_out.mean():.4f}  std={scores_out.std():.4f}")
    print(f"grid_points(m)={args.m}  gamma={args.gamma}  delta={args.delta}")

    # Sign-flip so larger scores always mean "more likely to be IN".
    si, so = scores_in.copy(), scores_out.copy()
    if si.mean() < so.mean():
        si, so = -si, -so

    method_order = [
        'Pointwise CP', 'Pointwise CP + symmetry',
        'Hull (no CP) [Alg5 no CP]', 'Hull (with CP) [CP/Alg5]',
        'GDP',
    ]

    def _run_combo(fit_si, fit_so, hold_si, hold_so, gamma_hold=None):
        if gamma_hold is None:
            gamma_hold = args.gamma
        fit_results = _select_fit_thresholds(fit_si, fit_so, args.delta, args.gamma, args.m)
        holdout_results = {}
        for label, fit_result in fit_results.items():
            if label == 'GDP':
                holdout_results[label] = _evaluate_threshold_gdp(
                    hold_si, hold_so, fit_result['threshold'], delta=args.delta, gamma=gamma_hold,
                )
            else:
                holdout_apply_cp = label != 'Hull (no CP) [Alg5 no CP]'
                holdout_results[label] = _evaluate_threshold(
                    hold_si, hold_so, fit_result['threshold'],
                    delta=args.delta, gamma=gamma_hold, apply_cp=holdout_apply_cp,
                )
        return fit_results, holdout_results

    def _run_kfold(scores_in, scores_out, use_lr):
        """K-fold CV: select threshold on k-1 folds, evaluate on kth with gamma/k (Bonferroni)."""
        k = args.n_folds
        gamma_hold = args.gamma / k
        splits = _kfold_splits(scores_in, scores_out, k, args.seed)
        fold_eps = {m: [] for m in method_order}
        for fit_si, fit_so, hold_si, hold_so in splits:
            if use_lr:
                fit_si, fit_so, hold_si, hold_so = apply_gaussian_lr_transform(
                    fit_si, fit_so, hold_si, hold_so)
            _, holdout_results = _run_combo(fit_si, fit_so, hold_si, hold_so, gamma_hold=gamma_hold)
            for m in method_order:
                fold_eps[m].append(holdout_results[m]['eps'])
        return {m: float(np.max(fold_eps[m])) for m in method_order}

    if args.all_combos:
        score_variants = [('raw', False), ('LR', True)]
        cols = [(sname, frac) for sname, _ in score_variants for frac in args.holdout_fracs]
        if args.n_folds > 1:
            cols += [(f'kfold(k={args.n_folds})', sname) for sname, _ in score_variants]
        col_header = '  '.join(f"{str(c):>12}" for c in [f'{s}/h={f}' if isinstance(f, float) else f'{f}/{s}'
                                                          for s, f in cols])
        print(f"\n{'Method':32s}  {col_header}")
        print('-' * (32 + 2 + len(col_header) + 2))

        for method in method_order:
            row_vals = []
            for sname, use_lr in score_variants:
                for frac in args.holdout_fracs:
                    fit_si, fit_so, hold_si, hold_so = _split_scores_for_holdout(si, so, frac, args.seed)
                    if use_lr:
                        fit_si, fit_so, hold_si, hold_so = apply_gaussian_lr_transform(
                            fit_si, fit_so, hold_si, hold_so)
                    _, holdout_results = _run_combo(fit_si, fit_so, hold_si, hold_so)
                    row_vals.append(holdout_results[method]['eps'])
            if args.n_folds > 1:
                for _, use_lr in score_variants:
                    kfold_eps = _run_kfold(si, so, use_lr)
                    row_vals.append(kfold_eps[method])
            vals_str = '  '.join(f"{v:12.4f}" for v in row_vals)
            print(f"{method:32s}  {vals_str}")

    else:
        score_label = 'Gaussian LR score' if args.use_lr else 'raw score'
        print(f"threshold sweep on: {score_label}")

        for holdout_frac in args.holdout_fracs:
            fit_si, fit_so, hold_si, hold_so = _split_scores_for_holdout(si, so, holdout_frac, args.seed)
            if args.use_lr:
                fit_si, fit_so, hold_si, hold_so = apply_gaussian_lr_transform(fit_si, fit_so, hold_si, hold_so)

            print(f"\n--- holdout_frac={holdout_frac}  fit={len(fit_si)}/side  holdout={len(hold_si)}/side ---")
            print(f"{'Method':30s}  {'fit_t':>10}  {'hold_eps':>8}  {'hold_a':>10}  {'hold_b':>10}  {'side':>6}")
            print('-' * 92)

            fit_results, holdout_results = _run_combo(fit_si, fit_so, hold_si, hold_so)
            for label in method_order:
                fit_t = fit_results[label]['threshold']
                hold = holdout_results[label]
                fit_t_str = f"{fit_t:.4f}" if fit_t is not None else "n/a"
                hold_a_str = f"{hold['alpha']:.4f}" if hold['alpha'] is not None else "n/a"
                hold_b_str = f"{hold['beta']:.4f}" if hold['beta'] is not None else "n/a"
                print(
                    f"{label:30s}  {fit_t_str:>10}  {hold['eps']:8.4f}  "
                    f"{hold_a_str:>10}  {hold_b_str:>10}  {hold['winner']:>6}"
                )

        if args.n_folds > 1:
            k = args.n_folds
            gamma_hold = args.gamma / k
            print(f"\n--- k-fold  k={k}  gamma/k={gamma_hold:.4f}  fit={(len(si) * (k-1)) // k}/side  holdout=~{len(si) // k}/side ---")
            print(f"{'Method':30s}  {'max_fold_eps':>12}")
            print('-' * 46)
            kfold_eps = _run_kfold(si, so, args.use_lr)
            for label in method_order:
                print(f"{label:30s}  {kfold_eps[label]:12.4f}")

    if args.print_grid:
        fit_si, fit_so, _, _ = _split_scores_for_holdout(si, so, args.holdout_fracs[0], args.seed)
        if args.use_lr or args.all_combos:
            fit_si, fit_so, _, _ = apply_gaussian_lr_transform(fit_si, fit_so, fit_si, fit_so)
        no_cp_grid = compute_eps_from_grid(fit_si, fit_so, args.delta, args.gamma, args.m, apply_cp=False)
        with_cp_grid = compute_eps_from_grid(fit_si, fit_so, args.delta, args.gamma, args.m, apply_cp=True)
        print(f"\nPer-grid points on fit split (holdout_frac={args.holdout_fracs[0]}):")
        print(f"{'idx':>5}  {'t':>10}  {'alpha_raw':>10}  {'beta_raw':>10}  {'alpha_cp':>10}  {'beta_cp':>10}")
        print('-' * 72)
        for idx, (raw_pt, cp_pt) in enumerate(zip(no_cp_grid['grid_points'], with_cp_grid['grid_points'])):
            print(
                f"{idx:5d}  {raw_pt[2]:10.4f}  {raw_pt[0]:10.4f}  {raw_pt[1]:10.4f}  "
                f"{cp_pt[0]:10.4f}  {cp_pt[1]:10.4f}"
            )


if __name__ == '__main__':
    main()
