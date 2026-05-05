"""
Cross-run per-sample empirical epsilon: no-defense vs defense.

Takes two experiment directories (no_defense_dir, defense_dir), each containing
all_losses_in.npy, all_losses_out.npy (shape: n_reps x n_samples), and also
losses_in.npy / losses_out.npy (shape: n_reps, the canary's own losses).

What all_losses_in/out[:,i] actually measures
---------------------------------------------
parallel_audit_model.py trains n_reps 'in' models (with the canary) and n_reps
'out' models (without the canary), then evaluates the FULL training set on each.
So all_losses_in[:,i] = loss of training sample i when the CANARY is included;
all_losses_out[:,i] = loss of training sample i when the canary is NOT included.

For i != canary_idx: sample i is in training in BOTH worlds; eps_i measures
"how well can an attacker detect the canary's membership by watching sample i's
loss?" — a valid (but indirect) canary-membership signal.

For i == canary_idx: all_losses_in[:,canary_idx] = target_X's loss under 'in'
models (correct). BUT all_losses_out[:,canary_idx] is X_out[canary_idx]'s loss
(a DIFFERENT sample, always a training member in 'out' world). This comparison
is invalid and produces spurious eps. We fix this by replacing the canary column
in all_out with the negated losses_out.npy (which correctly stores target_X's
loss under 'out' models in -CE convention).

Primary question: for each method, is the defense problematic?

  The defense is PROBLEMATIC if — after filtering — any sample's loss signal
  achieves empirical eps greater than the canary's eps BEFORE the defense.

  For each method:
    eps_canary_no_defense  = canary's eps from no_defense_dir
    max_eps_defense        = max eps over ALL samples in defense_dir
    PROBLEMATIC if max_eps_defense > eps_canary_no_defense

The canary is at index -1 by default (last sample). Override with --canary_idx.

Parallelized across samples with multiprocessing.
"""

import argparse
import os
import sys
import time
import multiprocessing as mp
import numpy as np
from scipy.stats import beta as beta_dist, norm


# ---------------------------------------------------------------------------
# GDP vectorized eps
# ---------------------------------------------------------------------------

def _beta_bounds(tp, fp, fn, tn, alpha):
    """Clopper-Pearson upper bounds on FPR and FNR. Computed once, shared by GDP and CP."""
    P = tp + fn
    N = fp + tn
    with np.errstate(invalid='ignore'):
        fpr_upper = np.where(
            (fp < N) & (N > 0),
            beta_dist.ppf(1 - alpha, fp + 1, np.maximum(N - fp, 1e-12)),
            1.0,
        )
        fnr_upper = np.where(
            (fn < P) & (P > 0),
            beta_dist.ppf(1 - alpha, fn + 1, np.maximum(P - fn, 1e-12)),
            1.0,
        )
    return fpr_upper, fnr_upper


def _gdp_from_bounds(fpr_upper, fnr_upper, delta):
    """GDP eps from pre-computed CP bounds."""
    with np.errstate(invalid='ignore'):
        mu_l = norm.ppf(1 - fpr_upper) - norm.ppf(fnr_upper)
    mu_l = np.where(np.isfinite(mu_l), mu_l, 0.0)

    eps = np.zeros_like(mu_l)
    valid = mu_l > 0
    if not valid.any():
        return eps

    mu_v = mu_l[valid]
    lo = np.zeros_like(mu_v)
    hi = np.full_like(mu_v, 50.0)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        with np.errstate(over='ignore', invalid='ignore'):
            f = (norm.cdf(-mid / mu_v + mu_v / 2)
                 - np.exp(mid) * norm.cdf(-mid / mu_v - mu_v / 2)
                 - delta)
        f = np.where(np.isfinite(f), f, -1.0)
        lo = np.where(f > 0, mid, lo)
        hi = np.where(f > 0, hi, mid)

    eps[valid] = 0.5 * (lo + hi)
    return np.clip(eps, 0.0, None)


def _cp_from_bounds(fpr_upper, fnr_upper, delta):
    """CP eps from pre-computed CP bounds."""
    numerator = 1.0 - fnr_upper - delta
    with np.errstate(divide='ignore', invalid='ignore'):
        eps = np.where(
            (numerator > 0) & (fpr_upper > 0),
            np.log(numerator / np.maximum(fpr_upper, 1e-300)),
            0.0,
        )
    return np.clip(eps, 0.0, None)


# ---------------------------------------------------------------------------
# Core per-sample computation
# ---------------------------------------------------------------------------

def _sweep_best_both(s_in, s_out, alpha, delta):
    """Sweep thresholds + both directions; compute GDP and CP in one pass.

    Returns ((gdp_eps, gdp_t, gdp_dir), (cp_eps, cp_t, cp_dir)).
    beta_dist.ppf is called once per direction (shared between GDP and CP).
    """
    gdp_best = (0.0, None, +1.0)
    cp_best  = (0.0, None, +1.0)
    for direction in (+1.0, -1.0):
        si = direction * s_in
        so = direction * s_out
        ts = np.unique(np.concatenate([si, so]))
        si_s = np.sort(si)
        so_s = np.sort(so)
        tp = len(si) - np.searchsorted(si_s, ts, side='left')
        fp = len(so) - np.searchsorted(so_s, ts, side='left')
        fn = len(si) - tp
        tn = len(so) - fp

        fpr_upper, fnr_upper = _beta_bounds(
            tp.astype(np.float64), fp.astype(np.float64),
            fn.astype(np.float64), tn.astype(np.float64), alpha,
        )

        gdp_arr = _gdp_from_bounds(fpr_upper, fnr_upper, delta)
        idx = int(np.argmax(gdp_arr))
        if gdp_arr[idx] > gdp_best[0]:
            gdp_best = (float(gdp_arr[idx]), float(ts[idx]), float(direction))

        cp_arr = _cp_from_bounds(fpr_upper, fnr_upper, delta)
        idx = int(np.argmax(cp_arr))
        if cp_arr[idx] > cp_best[0]:
            cp_best = (float(cp_arr[idx]), float(ts[idx]), float(direction))

    return gdp_best, cp_best


def _eval_fixed_both(hi, ho, gdp_t, gdp_dir, cp_t, cp_dir, alpha, delta):
    """Evaluate fixed (direction, threshold) on holdout for both methods."""
    def _eval(best_t, best_dir, method_fn):
        if best_t is None:
            return 0.0
        sh = best_dir * hi
        so = best_dir * ho
        tp = np.array([float(np.sum(sh >= best_t))])
        fp = np.array([float(np.sum(so >= best_t))])
        fn = np.array([float(np.sum(sh < best_t))])
        tn = np.array([float(np.sum(so < best_t))])
        fpr_u, fnr_u = _beta_bounds(tp, fp, fn, tn, alpha)
        return float(method_fn(fpr_u, fnr_u, delta)[0])

    return _eval(gdp_t, gdp_dir, _gdp_from_bounds), _eval(cp_t, cp_dir, _cp_from_bounds)


def per_sample_eps_all(losses_in_i, losses_out_i, seed, alpha, delta):
    """Compute all 4 eps variants for one sample.

    Returns (gdp_nh, cp_nh, gdp_h, cp_h):
      gdp_nh / cp_nh : no-holdout (NOT valid lower bounds)
      gdp_h  / cp_h  : 50% holdout (valid lower bounds at level alpha)
    """
    n = len(losses_in_i)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_hold = max(1, min(n // 2, n - 1))
    hold_idx, fit_idx = perm[:n_hold], perm[n_hold:]

    fi, fo = losses_in_i[fit_idx], losses_out_i[fit_idx]
    hi, ho = losses_in_i[hold_idx], losses_out_i[hold_idx]

    (gdp_nh, _, _), (cp_nh, _, _) = _sweep_best_both(losses_in_i, losses_out_i, alpha, delta)
    (_, gdp_t, gdp_dir), (_, cp_t, cp_dir) = _sweep_best_both(fi, fo, alpha, delta)

    gdp_h, cp_h = _eval_fixed_both(hi, ho, gdp_t, gdp_dir, cp_t, cp_dir, alpha, delta)

    return gdp_nh, cp_nh, gdp_h, cp_h


# ---------------------------------------------------------------------------
# Worker pool
# ---------------------------------------------------------------------------

_G_IN = None
_G_OUT = None
_G_PARAMS = None


def _init_worker(in_arr, out_arr, seed, alpha, delta):
    global _G_IN, _G_OUT, _G_PARAMS
    _G_IN = in_arr
    _G_OUT = out_arr
    _G_PARAMS = (seed, alpha, delta)


def _worker_chunk(chunk_indices):
    seed, alpha, delta = _G_PARAMS
    out = np.zeros((len(chunk_indices), 4), dtype=np.float64)
    for k, i in enumerate(chunk_indices):
        out[k] = per_sample_eps_all(_G_IN[:, i], _G_OUT[:, i], seed, alpha, delta)
    return chunk_indices, out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _load_losses(data_dir):
    in_path  = os.path.join(data_dir, 'all_losses_in.npy')
    out_path = os.path.join(data_dir, 'all_losses_out.npy')
    if not os.path.exists(in_path) or not os.path.exists(out_path):
        sys.exit(f"Missing all_losses_{{in,out}}.npy in {data_dir}")

    def _to_2d(raw):
        # Saved with dtype=object (list of per-rep loss arrays); stack into 2D float.
        if raw.dtype == object:
            return np.stack([np.asarray(x, dtype=np.float64) for x in raw])
        return raw.astype(np.float64)

    all_in  = _to_2d(np.load(in_path,  allow_pickle=True))
    all_out = _to_2d(np.load(out_path, allow_pickle=True))
    if all_in.shape != all_out.shape:
        sys.exit(f"Shape mismatch in {data_dir}: in={all_in.shape}, out={all_out.shape}")
    return all_in, all_out


def _load_canary_losses(data_dir):
    """Load target_X's per-rep losses from losses_in/out.npy (stored as -CE)."""
    in_path  = os.path.join(data_dir, 'losses_in.npy')
    out_path = os.path.join(data_dir, 'losses_out.npy')
    if not os.path.exists(in_path) or not os.path.exists(out_path):
        return None, None
    return (np.load(in_path).astype(np.float64),
            np.load(out_path).astype(np.float64))


def _compute_eps(label, all_in, all_out, seed, eff_alpha, delta, n_workers, chunk_size):
    """Run per-sample eps computation; return (n_samples, 4) array."""
    n_reps, n_samples = all_in.shape
    chunks = [np.arange(i, min(i + chunk_size, n_samples))
              for i in range(0, n_samples, chunk_size)]
    eps_all = np.zeros((n_samples, 4), dtype=np.float64)

    print(f"\n--- {label}: {n_reps} reps x {n_samples} samples ---")
    t0 = time.time()

    if n_workers > 1 and len(chunks) > 1:
        with mp.Pool(
            n_workers,
            initializer=_init_worker,
            initargs=(all_in, all_out, seed, eff_alpha, delta),
        ) as pool:
            n_done = 0
            log_every = max(1, n_samples // 20)
            next_log = log_every
            for chunk_indices, out in pool.imap_unordered(_worker_chunk, chunks):
                eps_all[chunk_indices] = out
                n_done += len(chunk_indices)
                if n_done >= next_log or n_done == n_samples:
                    el = time.time() - t0
                    rate = n_done / max(el, 1e-6)
                    eta = (n_samples - n_done) / max(rate, 1e-6)
                    mx = eps_all[:n_done].max(axis=0)
                    print(f"  [{n_done}/{n_samples}] {el:.1f}s ({rate:.0f}/s, eta={eta:.0f}s) "
                          f"max gdp_nh={mx[0]:.4f} cp_nh={mx[1]:.4f} "
                          f"gdp_h={mx[2]:.4f} cp_h={mx[3]:.4f}",
                          flush=True)
                    next_log = n_done + log_every
    else:
        _init_worker(all_in, all_out, seed, eff_alpha, delta)
        for chunk in chunks:
            ci, out = _worker_chunk(chunk)
            eps_all[ci] = out

    print(f"  Done in {time.time() - t0:.1f}s")
    return eps_all


def _recompute_subset(in_arr, out_arr, indices, seed, alpha, delta):
    """Recompute per-sample eps for a subset of indices at a given alpha."""
    out = np.zeros((len(indices), 4), dtype=np.float64)
    for k, i in enumerate(indices):
        out[k] = per_sample_eps_all(in_arr[:, i], out_arr[:, i], seed, alpha, delta)
    return out


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('no_defense_dir',
                        help='Directory from the run WITHOUT the defense')
    parser.add_argument('defense_dir',
                        help='Directory from the run WITH the defense')
    parser.add_argument('--canary_idx', type=int, default=-1,
                        help='Canary index in the dataset (default -1: last sample)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='Per-sample significance level (default 0.1)')
    parser.add_argument('--delta', type=float, default=1e-5)
    parser.add_argument('--bonferroni', action='store_true',
                        help='Use alpha/n_samples for a uniform bound across all samples')
    parser.add_argument('--n_workers', type=int, default=None,
                        help='Worker processes (default: min(cpu_count, 32))')
    parser.add_argument('--chunk_size', type=int, default=200,
                        help='Samples per worker chunk (default 200)')
    parser.add_argument('--save', action='store_true',
                        help='Save per_sample_eps_*.npy into each directory')
    args = parser.parse_args()

    nd_in,  nd_out  = _load_losses(args.no_defense_dir)
    def_in, def_out = _load_losses(args.defense_dir)

    n_reps_nd,  n_samples_nd  = nd_in.shape
    n_reps_def, n_samples_def = def_in.shape

    if n_samples_nd != n_samples_def:
        sys.exit(f"Sample count mismatch: no_defense={n_samples_nd}, defense={n_samples_def}")

    n_samples = n_samples_nd
    canary_idx = int(args.canary_idx) % n_samples
    eff_alpha = (args.alpha / n_samples) if args.bonferroni else args.alpha

    # Fix canary column in all_out: all_losses_out[:,canary_idx] stores a DIFFERENT
    # training sample's loss (X_out[canary_idx]), not target_X's loss under 'out' models.
    # losses_out.npy contains the correct target_X losses but in -CE convention;
    # negate to match all_losses' positive-CE convention.
    nd_canary_in,   nd_canary_out  = _load_canary_losses(args.no_defense_dir)
    def_canary_in,  def_canary_out = _load_canary_losses(args.defense_dir)

    # Patch FIRST — canary_eps_*_full must be computed from the corrected arrays.
    if nd_canary_out is not None:
        if len(nd_canary_out) != n_reps_nd:
            sys.exit(f"losses_out.npy in {args.no_defense_dir} has {len(nd_canary_out)} reps "
                     f"but all_losses has {n_reps_nd}")
        nd_out[:, canary_idx] = -nd_canary_out  # -(-CE) = +CE, lower = member
    else:
        print(f"WARNING: losses_out.npy not found in {args.no_defense_dir}; "
              f"canary eps from no-defense run may be wrong.")

    if def_canary_out is not None:
        if len(def_canary_out) != n_reps_def:
            sys.exit(f"losses_out.npy in {args.defense_dir} has {len(def_canary_out)} reps "
                     f"but all_losses has {n_reps_def}")
        def_out[:, canary_idx] = -def_canary_out
    else:
        print(f"WARNING: losses_out.npy not found in {args.defense_dir}; "
              f"canary eps from defense run may be wrong.")

    # Canary eps at FULL alpha (not eff_alpha) — needed as the reference point in the
    # asymmetric Bonferroni cross-run check regardless of whether --bonferroni is set.
    # Computed AFTER the patch so nd_out[:,canary_idx] is target_X's 'out' losses.
    canary_eps_nd_full = np.array(
        per_sample_eps_all(nd_in[:, canary_idx], nd_out[:, canary_idx],
                           args.seed, args.alpha, args.delta),
        dtype=np.float64,
    )
    canary_eps_def_full = np.array(
        per_sample_eps_all(def_in[:, canary_idx], def_out[:, canary_idx],
                           args.seed, args.alpha, args.delta),
        dtype=np.float64,
    )
    n_non_canary = n_samples - 1
    bonf_alpha_check = args.alpha / max(n_non_canary, 1)

    print(f"no_defense_dir : {args.no_defense_dir}  ({n_reps_nd} reps)")
    print(f"defense_dir    : {args.defense_dir}  ({n_reps_def} reps)")
    print(f"samples        : {n_samples}   canary_idx: {canary_idx}")
    print(f"alpha={eff_alpha:.3e} ({'Bonferroni' if args.bonferroni else 'per-sample'}), "
          f"delta={args.delta}")
    print(f"Bonferroni alpha for cross-run check: {bonf_alpha_check:.3e}")

    n_workers  = args.n_workers or min(mp.cpu_count(), 32)
    chunk_size = max(1, int(args.chunk_size))

    eps_nd  = _compute_eps("NO DEFENSE",  nd_in,  nd_out,
                           args.seed, eff_alpha, args.delta, n_workers, chunk_size)
    eps_def = _compute_eps("WITH DEFENSE", def_in, def_out,
                           args.seed, eff_alpha, args.delta, n_workers, chunk_size)

    # column layout: [gdp_nh, cp_nh, gdp_h, cp_h]
    METHOD_NAMES = [
        'GDP no holdout  (inflated)',
        'CP  no holdout  (inflated)',
        'GDP 50% holdout (valid lb)',
        'CP  50% holdout (valid lb)',
    ]

    nc_mask = np.ones(n_samples, dtype=bool)
    nc_mask[canary_idx] = False

    print(f"\n{'='*72}")
    print("RESULTS")
    print('='*72)

    for col, name in enumerate(METHOD_NAMES):
        nd  = eps_nd[:, col]
        de  = eps_def[:, col]
        nc_de = de[nc_mask]

        # Canary reference at FULL alpha (single test, no selection)
        canary_nd_full = float(canary_eps_nd_full[col])
        canary_de_full = float(canary_eps_def_full[col])

        # Bonferroni recompute for BOTH no-defense and defense non-canary samples.
        # Efficiency: Bonferroni eps <= full-alpha eps, so eps=0 at full alpha stays 0.
        def _bonf_recompute(in_arr, out_arr, base_eps_col):
            mask = nc_mask & (base_eps_col > 0)
            idx  = np.where(mask)[0]
            if len(idx) == 0:
                return 0.0, -1, 0
            rec = _recompute_subset(in_arr, out_arr, idx,
                                    args.seed, bonf_alpha_check, args.delta)
            best = int(np.argmax(rec[:, col]))
            return float(rec[best, col]), int(idx[best]), len(idx)

        nc_max_bonf_nd,  nc_max_bonf_nd_idx,  n_recomp_nd  = _bonf_recompute(nd_in,  nd_out,  nd)
        nc_max_bonf_def, nc_max_bonf_def_idx, n_recomp_def = _bonf_recompute(def_in, def_out, de)

        # Distribution stats (at eff_alpha, for reference)
        max_de     = float(de.max())
        max_de_idx = int(np.argmax(de))
        nc_max_de  = float(nc_de.max())
        nc_max_idx = int(np.where(nc_mask)[0][np.argmax(nc_de)])

        problematic = nc_max_bonf_def > canary_nd_full
        verdict = "PROBLEMATIC" if problematic else "OK"

        print(f"\n{'='*72}")
        print(f"  {name}")
        print(f"{'='*72}")

        print(f"\n  No-defense distribution (α={eff_alpha:.2e}):")
        print(f"    canary eps:        {canary_nd_full:.6f}")
        print(f"    max (any sample):  {nd.max():.6f}  at idx {int(np.argmax(nd))}")
        print(f"    max non-canary:    {float(nd[nc_mask].max()):.6f}")
        print(f"    p99:               {np.percentile(nd, 99):.6f}")
        print(f"    p95:               {np.percentile(nd, 95):.6f}")
        print(f"    fraction > 0:      {(nd > 0).mean()*100:.2f}%")

        print(f"\n  Defense distribution (α={eff_alpha:.2e}):")
        print(f"    canary eps:        {canary_de_full:.6f}")
        print(f"    max (any sample):  {max_de:.6f}  at idx {max_de_idx}")
        print(f"    max non-canary:    {nc_max_de:.6f}  at idx {nc_max_idx}")
        print(f"    p99:               {np.percentile(de, 99):.6f}")
        print(f"    p95:               {np.percentile(de, 95):.6f}")
        print(f"    fraction > 0:      {(de > 0).mean()*100:.2f}%")

        print(f"\n  Cross-run check (Bonferroni α={bonf_alpha_check:.2e} per non-canary test):")
        print(f"    canary eps BEFORE defense (full α):  {canary_nd_full:.6f}")
        print(f"    max non-canary BEFORE defense:       {nc_max_bonf_nd:.6f}"
              f"  at idx {nc_max_bonf_nd_idx}"
              f"  ({n_recomp_nd} recomputed)")
        print(f"    max non-canary AFTER  defense:       {nc_max_bonf_def:.6f}"
              f"  at idx {nc_max_bonf_def_idx}"
              f"  ({n_recomp_def} recomputed)")
        print(f"    defense {verdict}  "
              f"(max after {'>' if problematic else '<='} canary before)")


    if args.save:
        for data_dir, eps_arr, tag in [
            (args.no_defense_dir, eps_nd,  'no_defense'),
            (args.defense_dir,    eps_def, 'defense'),
        ]:
            for col, method in [('gdp_no_holdout', 0), ('cp_no_holdout', 1),
                                 ('gdp_holdout', 2),    ('cp_holdout', 3)]:
                p = os.path.join(data_dir, f'per_sample_eps_{col}.npy')
                np.save(p, eps_arr[:, method])
                print(f"Saved {p}")


if __name__ == '__main__':
    main()
