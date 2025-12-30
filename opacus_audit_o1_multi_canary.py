import argparse
import os
import time
import math

import dill
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import root_scalar
from scipy.stats import norm

from models import Models
from utils.data import load_data
from parallel_audit_multi_canary import train_model_multi_canary


def _make_canaries_blank(X_ref: torch.Tensor, y_ref: torch.Tensor, n_canaries: int, blank_alpha: float):
    """Create `n_canaries` blank canaries matching X_ref shape.

    blank_alpha interpolates between a zero image and the original reference sample:
        x_canary = (1-blank_alpha)*0 + blank_alpha*x_ref
    """
    if not (0.0 <= float(blank_alpha) <= 1.0):
        raise ValueError(f"blank_alpha must be in [0, 1], got {blank_alpha}")

    if X_ref.ndim == 3:
        base = X_ref[-1].unsqueeze(0)
    else:
        base = X_ref[-1]
        if base.ndim == X_ref.ndim - 1:
            base = base.unsqueeze(0)

    blank = torch.zeros_like(base)
    x = (1.0 - blank_alpha) * blank + blank_alpha * base

    # Use same label for all blank canaries (match ref label)
    if y_ref.ndim == 0:
        y_base = y_ref.view(1)
    else:
        y_base = y_ref[-1].view(1)

    Xc = x.repeat(n_canaries, *([1] * (x.ndim - 1)))
    yc = y_base.repeat(n_canaries)
    return Xc, yc


def _load_canaries_from_pt_dict(pt_path: str, ref_X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Load multiple canaries from a single .pt file.

    Supported schemas:
    1) {'canaries': Tensor[N,...], 'audit_labels': Tensor[N] | list[int]}
    2) {'canaries': [{'canary': Tensor[...], 'audit_label': int}, ...]}
    3) {'canary': Tensor[...], 'audit_label': int}  (single canary)
    """
    if pt_path is None:
        raise ValueError("pt_path must not be None")
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"Canary pt file not found: {pt_path}")

    d = torch.load(pt_path, map_location='cpu')
    if not isinstance(d, dict):
        raise ValueError(f"Expected {pt_path} to be a dict")

    if 'canaries' in d and 'audit_labels' in d:
        X_canary = d['canaries']
        y_canary = d['audit_labels']
        if not torch.is_tensor(X_canary):
            raise ValueError("Expected 'canaries' to be a Tensor")
        if torch.is_tensor(y_canary):
            y_canary = y_canary.detach().cpu().tolist()
        if not isinstance(y_canary, (list, tuple)):
            raise ValueError("Expected 'audit_labels' to be a Tensor or list")
        y_canary = torch.tensor([int(x) for x in y_canary], dtype=torch.long)
    elif 'canaries' in d and isinstance(d['canaries'], (list, tuple)):
        items = d['canaries']
        Xs = []
        ys = []
        for item in items:
            if not isinstance(item, dict) or 'canary' not in item or 'audit_label' not in item:
                raise ValueError("Expected each entry in 'canaries' to be a dict with keys 'canary' and 'audit_label'")
            x = item['canary']
            if not torch.is_tensor(x):
                raise ValueError("Expected item['canary'] to be a Tensor")
            y = int(item['audit_label'])
            if x.ndim == ref_X.ndim - 1:
                x = x.unsqueeze(0)
            Xs.append(x)
            ys.append(y)
        X_canary = torch.cat(Xs, dim=0)
        y_canary = torch.tensor(ys, dtype=torch.long)
    elif 'canary' in d and 'audit_label' in d:
        x = d['canary']
        if not torch.is_tensor(x):
            raise ValueError("Expected 'canary' to be a Tensor")
        if x.ndim == ref_X.ndim - 1:
            x = x.unsqueeze(0)
        X_canary = x
        y_canary = torch.tensor([int(d['audit_label'])], dtype=torch.long)
    else:
        raise ValueError(
            "Unrecognized canary .pt schema. Supported schemas:\n"
            "  1) {'canaries': Tensor[N,...], 'audit_labels': Tensor[N]|list[int]}\n"
            "  2) {'canaries': [{'canary': Tensor, 'audit_label': int}, ...]}\n"
            "  3) {'canary': Tensor, 'audit_label': int}"
        )

    if X_canary.ndim == ref_X.ndim - 1:
        X_canary = X_canary.unsqueeze(0)
    if X_canary.ndim != ref_X.ndim:
        raise ValueError(f"Loaded canaries have ndim={X_canary.ndim}, expected {ref_X.ndim}")
    if tuple(X_canary.shape[1:]) != tuple(ref_X.shape[1:]):
        raise ValueError(f"Loaded canaries have sample shape {tuple(X_canary.shape[1:])}, expected {tuple(ref_X.shape[1:])}")
    if y_canary.ndim != 1 or y_canary.shape[0] != X_canary.shape[0]:
        raise ValueError(f"audit_labels must have shape (N,), got {tuple(y_canary.shape)} for N={X_canary.shape[0]}")

    meta = {
        'canary_pt_path': pt_path,
        'n_canaries_loaded': int(X_canary.shape[0]),
    }
    return X_canary, y_canary, meta


def compute_T_from_scores(scores: np.ndarray, k_plus: int, k_minus: int) -> np.ndarray:
    """Given per-canary scores, output T in {-1,0,+1} with abstention."""
    m = int(scores.shape[0])
    if k_plus < 0 or k_minus < 0 or (k_plus + k_minus) > m:
        raise ValueError(f"Invalid k_plus/k_minus: k_plus={k_plus} k_minus={k_minus} m={m}")

    order = np.argsort(scores)  # ascending
    T = np.zeros(m, dtype=np.int8)

    if k_minus > 0:
        T[order[:k_minus]] = -1
    if k_plus > 0:
        T[order[-k_plus:]] = 1

    return T


def compute_W(S: np.ndarray, T: np.ndarray) -> int:
    """W := sum_i max(0, T_i * S_i)."""
    prod = (T.astype(np.int32) * S.astype(np.int32))
    return int(np.maximum(0, prod).sum())


def _logsumexp(log_values: list[float]) -> float:
    if len(log_values) == 0:
        return float('-inf')
    m = max(log_values)
    if not math.isfinite(m):
        return m
    s = 0.0
    for v in log_values:
        s += math.exp(v - m)
    return m + math.log(s)


def _binom_tail_geq(r: int, q: float, v: int) -> float:
    if v <= 0:
        return 1.0
    if v > r:
        return 0.0
    q = float(q)
    if q <= 0.0:
        return 0.0
    if q >= 1.0:
        return 1.0

    log_q = math.log(q)
    log_1mq = math.log1p(-q)
    logs: list[float] = []
    for k in range(v, r + 1):
        log_c = math.lgamma(r + 1) - math.lgamma(k + 1) - math.lgamma(r - k + 1)
        logs.append(log_c + k * log_q + (r - k) * log_1mq)
    return float(math.exp(_logsumexp(logs)))


def _binom_range_prob(r: int, q: float, lo: int, hi: int) -> float:
    if lo > hi:
        return 0.0
    lo = max(int(lo), 0)
    hi = min(int(hi), int(r))
    if lo > hi:
        return 0.0
    q = float(q)
    if q <= 0.0:
        return 1.0 if lo <= 0 <= hi else 0.0
    if q >= 1.0:
        return 1.0 if lo <= r <= hi else 0.0

    log_q = math.log(q)
    log_1mq = math.log1p(-q)
    logs: list[float] = []
    for k in range(lo, hi + 1):
        log_c = math.lgamma(r + 1) - math.lgamma(k + 1) - math.lgamma(r - k + 1)
        logs.append(log_c + k * log_q + (r - k) * log_1mq)
    return float(math.exp(_logsumexp(logs)))


def p_value_dp_audit(*, m: int, r: int, v: int, eps: float, delta: float) -> float:
    m = int(m)
    r = int(r)
    v = int(v)
    if m <= 0:
        raise ValueError(f"m must be > 0, got {m}")
    if r < 0 or r > m:
        raise ValueError(f"r must be in [0, m], got r={r} m={m}")
    if v < 0 or v > r:
        raise ValueError(f"v must be in [0, r], got v={v} r={r}")

    eps = float(eps)
    delta = float(delta)
    if eps < 0:
        raise ValueError(f"eps must be >= 0, got {eps}")
    if delta < 0:
        raise ValueError(f"delta must be >= 0, got {delta}")

    q = 1.0 / (1.0 + math.exp(-eps))
    beta = _binom_tail_geq(r=r, q=q, v=v)

    alpha = 0.0
    for i in range(1, v + 1):
        prob = _binom_range_prob(r=r, q=q, lo=v - i, hi=v - 1)
        alpha = max(alpha, prob / float(i))

    p = float(beta) + float(alpha) * float(delta) * 2.0 * float(m)
    return float(min(p, 1.0))


def get_eps_audit(*, m: int, r: int, v: int, delta: float, alpha: float, n_iter: int = 30) -> float:
    m = int(m)
    r = int(r)
    v = int(v)
    delta = float(delta)
    alpha = float(alpha)
    if alpha <= 0.0 or alpha >= 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    eps_min = 0.0
    eps_max = 1.0
    while p_value_dp_audit(m=m, r=r, v=v, eps=eps_max, delta=delta) < alpha:
        eps_max *= 2.0
        if eps_max > 1024.0:
            break

    for _ in range(int(n_iter)):
        eps = 0.5 * (eps_min + eps_max)
        if p_value_dp_audit(m=m, r=r, v=v, eps=eps, delta=delta) < alpha:
            eps_min = eps
        else:
            eps_max = eps

    return float(eps_min)


def _gaussian_tradeoff_inverse(noise_multiplier: float):
    nm = float(noise_multiplier)
    if nm <= 0.0:
        raise ValueError(f"noise_multiplier must be > 0, got {noise_multiplier}")

    def f_inv(x: float) -> float:
        x = float(x)
        # Guard numeric extremes to avoid inf-inf issues.
        if x <= 0.0:
            return 0.0
        if x >= 1.0:
            return 1.0
        return float(norm.cdf(norm.ppf(x) - 1.0 / nm))

    return f_inv


def evaluate_f_dp(
    *,
    inverse_tradeoff_func,
    m: int,
    c: int,
    c_prime: int,
    k: int = 2,
    tau: float = 0.05,
) -> bool:
    m = int(m)
    c = int(c)
    c_prime = int(c_prime)
    k = int(k)
    tau = float(tau)

    if m <= 0:
        raise ValueError(f"m must be > 0, got {m}")
    if c_prime < 0 or c_prime > m:
        raise ValueError(f"c_prime must be in [0, m], got c_prime={c_prime} m={m}")
    if c < 0 or c > c_prime:
        raise ValueError(f"c must be in [0, c_prime], got c={c} c_prime={c_prime}")
    if k < 2:
        raise ValueError(f"k must be >= 2, got {k}")
    if tau <= 0.0 or tau >= 1.0:
        raise ValueError(f"tau must be in (0, 1), got {tau}")

    if c_prime == 0:
        # Adversary abstained on everything; accept any hypothesis.
        return True

    adjusted_tau = tau * (float(c_prime) / float(m))

    h = [0.0] * (c + 1)
    r = [0.0] * (c + 1)

    r[c] = adjusted_tau * (float(c) / float(c_prime))
    h[c] = adjusted_tau * (float(c_prime - c) / float(c_prime))

    for i in range(c - 1, -1, -1):
        h[i] = max(float(h[i + 1]), float(k - 1) * float(inverse_tradeoff_func(r[i + 1])))

        denom = float(c_prime - i)
        if denom <= 0.0:
            r[i] = r[i + 1]
        else:
            r[i] = r[i + 1] + (float(i) / denom) * (h[i] - h[i + 1])

    return (float(r[0]) + float(h[0])) <= (float(c_prime) / float(m))


def _eps_from_mu_gdp(*, mu: float, delta: float) -> float:
    mu = float(mu)
    delta = float(delta)
    if mu <= 0.0:
        return 0.0

    def eq6(epsilon: float) -> float:
        return float(norm.cdf(-epsilon / mu + mu / 2.0) - math.exp(epsilon) * norm.cdf(-epsilon / mu - mu / 2.0) - delta)

    try:
        sol = root_scalar(eq6, bracket=[0.0, 50.0], method='brentq')
        return float(sol.root)
    except Exception:
        return 0.0


def get_empirical_epsilon_fdp_gaussian(
    *,
    m: int,
    c: int,
    c_prime: int,
    delta: float,
    tau: float,
    candidate_noises: np.ndarray,
) -> tuple[float, float]:
    strongest_valid_noise = 0.0
    candidate_noises = np.asarray(candidate_noises, dtype=np.float64)
    if candidate_noises.ndim != 1 or candidate_noises.size == 0:
        raise ValueError("candidate_noises must be a non-empty 1D array")

    # Expect sorted high -> low (most private -> least private)
    for noise in candidate_noises:
        f_inv = _gaussian_tradeoff_inverse(float(noise))
        ok = evaluate_f_dp(inverse_tradeoff_func=f_inv, m=int(m), c=int(c), c_prime=int(c_prime), k=2, tau=float(tau))
        if ok:
            strongest_valid_noise = float(noise)
        else:
            break

    if strongest_valid_noise <= 0.0:
        return 0.0, 0.0

    mu = 1.0 / strongest_valid_noise
    eps = _eps_from_mu_gdp(mu=mu, delta=float(delta))
    return float(eps), float(strongest_valid_noise)


def main():
    parser = argparse.ArgumentParser(description='O(1)-run multi-canary audit using custom DP-SGD training')

    parser.add_argument('--data_name', type=str, default='mnist', help='dataset to use (mnist, cifar10, cifar100)')
    parser.add_argument('--model_name', type=str, default='cnn', choices=list(Models.keys()), help='model to audit')

    parser.add_argument('--n_df', type=int, default=0, help='|D| (0 => use full dataset)')
    parser.add_argument('--n_canaries', type=int, default=5, help='number of canaries (auditing examples)')

    parser.add_argument('--k_plus', type=int, default=1, help='number of top-scoring canaries guessed IN')
    parser.add_argument('--k_minus', type=int, default=1, help='number of bottom-scoring canaries guessed OUT')

    parser.add_argument('--target_type', type=str, default='blank', help='canary type (blank or filelist)')
    parser.add_argument('--canary_pt', type=str, default=None,
                        help='path to a .pt dict containing canaries + labels; used when --target_type=pt')
    parser.add_argument('--blank_alpha', type=float, default=0.0, help='interpolation factor for blank target')

    parser.add_argument('--seed', type=int, default=0, help='seed for reproducibility')
    parser.add_argument('--out', type=str, default='debug/o1_audit', help='output folder')

    parser.add_argument('--n_epochs', type=int, default=10, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=1.33e-4, help='learning rate')
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam'], help='optimizer to use')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size for training')
    parser.add_argument('--block_size', type=int, default=256, help='block size for per-sample grad processing (default: batch_size)')

    parser.add_argument('--epsilon', type=float, default=10.0, help='privacy parameter, epsilon')
    parser.add_argument('--delta', type=float, default=1e-5, help='privacy parameter, delta')
    parser.add_argument('--alpha', type=float, default=0.05, help='significance level for empirical epsilon audit')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='gradient clipping norm (-1 => non-private)')

    parser.add_argument('--empirical_eps_method', type=str, default='pvalue', choices=['pvalue', 'fdp_gaussian'],
                        help='method used to compute empirical epsilon (default: pvalue)')
    parser.add_argument('--fdp_noise_max', type=float, default=50.0,
                        help='max candidate Gaussian noise multiplier for fdp_gaussian search (default: 50)')
    parser.add_argument('--fdp_noise_min', type=float, default=0.1,
                        help='min candidate Gaussian noise multiplier for fdp_gaussian search (default: 0.1)')
    parser.add_argument('--fdp_noise_steps', type=int, default=200,
                        help='number of candidate noise values (log-spaced) for fdp_gaussian (default: 200)')

    parser.add_argument('--aug_mult', type=int, default=1, help='augmentation multiplier (default: 1)')

    parser.add_argument('--defense', action='store_true', help='use filtering defense during training')
    parser.add_argument('--defense_k', type=int, default=5,
                        help='number of top samples to mark per class per epoch (default: 5)')
    parser.add_argument('--defense_apply_ascent', action='store_true', default=True,
                        help='if set, apply gradient ascent before dropping (default: True)')
    parser.add_argument('--defense_score_norm', type=str, default='linf', choices=['linf', 'l2', 'l1'],
                        help='norm used for defense score computation (default: linf)')
    parser.add_argument('--defense_score_fn', type=str, default='grad_norm',
                        help='defense scoring function name (default: grad_norm)')

    args = parser.parse_args()

    if args.optimizer != 'sgd':
        raise ValueError("Only --optimizer=sgd is supported with the custom training loop")

    if args.n_canaries < 1:
        raise ValueError(f"--n_canaries must be >= 1, got {args.n_canaries}")

    if not (0.0 < float(args.alpha) < 1.0):
        raise ValueError(f"--alpha must be in (0, 1), got {args.alpha}")

    if float(args.fdp_noise_min) <= 0.0 or float(args.fdp_noise_max) <= 0.0:
        raise ValueError("--fdp_noise_min and --fdp_noise_max must be > 0")
    if float(args.fdp_noise_min) > float(args.fdp_noise_max):
        raise ValueError(f"Expected --fdp_noise_min <= --fdp_noise_max, got {args.fdp_noise_min} > {args.fdp_noise_max}")
    if int(args.fdp_noise_steps) < 2:
        raise ValueError(f"--fdp_noise_steps must be >= 2, got {args.fdp_noise_steps}")

    if not (0.0 <= float(args.blank_alpha) <= 1.0):
        raise ValueError(f"--blank_alpha must be in [0, 1], got {args.blank_alpha}")

    if args.epsilon == -1:
        args.epsilon = None

    if args.max_grad_norm == -1:
        args.max_grad_norm = None

    # Keep privacy knobs consistent: either both are set (private) or both are None (non-private).
    if args.epsilon is None or args.max_grad_norm is None:
        args.epsilon = None
        args.max_grad_norm = None

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.out, exist_ok=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset
    if args.n_df == 1:
        X, y, out_dim = load_data(args.data_name, 1, split='train')
    else:
        X, y, out_dim = load_data(args.data_name, args.n_df - 1, split='train')
    y = y.long()

    n = len(X)
    m = int(args.n_canaries)

    if n <= m:
        raise ValueError(f"Need n_df > n_canaries. Got n={n} m={m}")

    # Construct canaries (auditing examples) + non-auditing base
    canary_meta = {}
    if args.target_type == 'blank':
        X_canary, y_canary = _make_canaries_blank(X, y, n_canaries=m, blank_alpha=float(args.blank_alpha))
    elif args.target_type == 'pt':
        if args.canary_pt is None:
            raise ValueError("--canary_pt is required when --target_type=pt")
        X_canary, y_canary, canary_meta = _load_canaries_from_pt_dict(args.canary_pt, ref_X=X)
        m = int(X_canary.shape[0])
        if m != int(args.n_canaries):
            raise ValueError(f"Loaded {m} canaries from pt, but --n_canaries was {args.n_canaries}")
    else:
        raise ValueError(f"Unsupported --target_type {args.target_type} (expected blank or pt)")

    # Non-auditing examples: always included
    X_base = X[:-m]
    y_base = y[:-m]

    # Sample inclusion mask S for canaries: +1 included, -1 excluded (coin flips)
    S = rng.integers(low=0, high=2, size=m, dtype=np.int64)
    S = np.where(S == 1, 1, -1).astype(np.int8)

    include_mask = (S == 1)
    X_in = torch.vstack([X_base, X_canary[include_mask]])
    y_in = torch.cat([y_base, y_canary[include_mask]])

    included_canary_positions = np.where(include_mask)[0].astype(np.int64)
    canary_indices_in_train = None
    if args.defense:
        # The included canaries are appended to the end of X_in.
        base_len = int(X_base.shape[0])
        n_included = int(include_mask.sum())
        canary_indices_in_train = np.arange(base_len, base_len + n_included, dtype=np.int64)

    print(f"Base size={len(X_base)} canaries={m} included={int(include_mask.sum())} excluded={int((~include_mask).sum())}")
    print(f"Training set size={len(X_in)}")

    gen = torch.Generator(device='cpu')
    gen.manual_seed(int(args.seed) + 1)
    dl_gen = torch.Generator(device='cpu')
    dl_gen.manual_seed(int(args.seed) + 2)

    # Train once
    start = time.time()
    # Note: train_model_multi_canary is the copied training loop derived from parallel_audit_model.
    # It supports multi-canary defense drop tracking via canary_indices.
    model, _drop_mask, defense_stats = train_model_multi_canary(
        model_name=args.model_name,
        X=X_in,
        y=y_in,
        epsilon=args.epsilon,
        delta=args.delta,
        max_grad_norm=args.max_grad_norm,
        n_epochs=args.n_epochs,
        lr=args.lr,
        block_size=min(int(args.block_size), int(args.batch_size)),
        batch_size=min(int(args.batch_size), int(X_in.shape[0])),
        init_model=None,
        out_dim=out_dim,
        aug_mult=args.aug_mult,
        defense=bool(args.defense),
        defense_k=args.defense_k,
        defense_apply_ascent=bool(args.defense_apply_ascent),
        defense_filter_every=1,
        defense_score_fn=str(args.defense_score_fn),
        defense_score_norm=str(args.defense_score_norm),
        device=str(device),
        generator=gen,
        dl_generator=dl_gen,
        num_workers=0,
        persistent_workers=False,
        canary_indices=canary_indices_in_train,
    )
    print(f"Training done in {time.time() - start:.2f}s")

    # Score all canaries with final model
    model.eval()
    with torch.no_grad():
        dev = next(model.parameters()).device
        logits = model(X_canary.to(dev))
        scores = (-F.cross_entropy(logits, y_canary.to(dev), reduction='none')).detach().cpu().numpy().astype(np.float32)

    # Guess T with abstention
    T = compute_T_from_scores(scores, args.k_plus, args.k_minus)

    # Compute W
    W = compute_W(S, T)

    # Useful summary stats on guessed items only
    guessed = (T != 0)
    n_guessed = int(guessed.sum())
    n_correct = int(((T[guessed] * S[guessed]) > 0).sum()) if n_guessed > 0 else 0

    emp_eps = None
    emp_eps_aux = None
    if str(args.empirical_eps_method) == 'pvalue':
        emp_eps = get_eps_audit(
            m=m,
            r=n_guessed,
            v=n_correct,
            delta=float(args.delta),
            alpha=float(args.alpha),
            n_iter=30,
        )
    elif str(args.empirical_eps_method) == 'fdp_gaussian':
        noises = np.logspace(
            np.log10(float(args.fdp_noise_max)),
            np.log10(float(args.fdp_noise_min)),
            num=int(args.fdp_noise_steps),
            base=10.0,
            dtype=np.float64,
        )
        emp_eps, emp_eps_aux = get_empirical_epsilon_fdp_gaussian(
            m=m,
            c=n_correct,
            c_prime=n_guessed,
            delta=float(args.delta),
            tau=float(args.alpha),
            candidate_noises=noises,
        )
    else:
        raise ValueError(f"Unknown --empirical_eps_method: {args.empirical_eps_method}")

    print(f"k_plus={args.k_plus} k_minus={args.k_minus} guessed={n_guessed}/{m} correct={n_correct}/{n_guessed}")
    print(f"W={W}")
    print(f"Empirical epsilon (alpha={args.alpha}, delta={args.delta}): {emp_eps}")

    # Save outputs
    np.save(os.path.join(args.out, 'S.npy'), S)
    np.save(os.path.join(args.out, 'T.npy'), T)
    np.save(os.path.join(args.out, 'scores.npy'), scores)
    np.save(os.path.join(args.out, 'W.npy'), np.asarray(W, dtype=np.int32))
    np.save(os.path.join(args.out, 'emp_eps.npy'), np.asarray(emp_eps, dtype=np.float32))

    if args.defense:
        # Align canary drop epochs back to the original canary ids (0..m-1)
        # -2: excluded canary (not in training)
        # -1: included but not dropped
        canary_drop_epochs_aligned = np.full(m, -2, dtype=np.int32)
        canary_drop_ratio_events_by_epoch = []
        if defense_stats is not None and defense_stats.get('canary_drop_epochs') is not None:
            drop_epochs_included = np.asarray(defense_stats['canary_drop_epochs'], dtype=np.int32)
            if drop_epochs_included.shape[0] != int(include_mask.sum()):
                raise ValueError(
                    f"Defense returned drop epochs for {drop_epochs_included.shape[0]} canaries, "
                    f"but include_mask has {int(include_mask.sum())} included canaries"
                )
            canary_drop_epochs_aligned[included_canary_positions] = drop_epochs_included
            canary_drop_ratio_events_by_epoch = defense_stats.get('canary_drop_ratio_events') or []
        else:
            canary_drop_epochs_aligned[included_canary_positions] = -1

        np.save(os.path.join(args.out, 'canary_drop_epochs.npy'), canary_drop_epochs_aligned)
        dill.dump(
            {
                'included_canary_positions': included_canary_positions,
                'canary_drop_ratio_events': canary_drop_ratio_events_by_epoch,
            },
            open(os.path.join(args.out, 'defense_stats.dill'), 'wb')
        )

    meta = {
        'data_name': args.data_name,
        'model_name': args.model_name,
        'n_df': int(args.n_df),
        'n_canaries': int(m),
        'k_plus': int(args.k_plus),
        'k_minus': int(args.k_minus),
        'target_type': args.target_type,
        'epsilon': args.epsilon,
        'delta': float(args.delta),
        'alpha': float(args.alpha),
        'max_grad_norm': args.max_grad_norm,
        'batch_size': int(args.batch_size),
        'n_epochs': int(args.n_epochs),
        'lr': float(args.lr),
        'optimizer': args.optimizer,
        'aug_mult': int(args.aug_mult),
        'blank_alpha': float(args.blank_alpha),
        'seed': int(args.seed),
        'defense': bool(args.defense),
        'defense_k': int(args.defense_k),
        'defense_apply_ascent': bool(args.defense_apply_ascent),
        'defense_score_fn': str(args.defense_score_fn),
        'defense_score_norm': str(args.defense_score_norm),
        'emp_eps': float(emp_eps),
        'empirical_eps_method': str(args.empirical_eps_method),
        'fdp_strongest_valid_noise': float(emp_eps_aux) if emp_eps_aux is not None else None,
        'fdp_noise_min': float(args.fdp_noise_min),
        'fdp_noise_max': float(args.fdp_noise_max),
        'fdp_noise_steps': int(args.fdp_noise_steps),
    }
    meta.update(canary_meta)
    dill.dump(meta, open(os.path.join(args.out, 'meta.dill'), 'wb'))


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nInterrupted by user')
    except Exception as e:
        print(f'\nError: {str(e)}')
        import traceback
        traceback.print_exc()
