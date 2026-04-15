"""
RDP-based privacy accounting for DP-SGD.

Computes the Gaussian noise multiplier (σ) required to achieve a target
(ε, δ)-DP guarantee using Rényi Differential Privacy composition.

Reference:
    Wang et al. (2019), "Subsampled Rényi Differential Privacy and
    Analytical Moments Accountant". https://arxiv.org/abs/1908.08765
"""
import numpy as np
from scipy.special import gammaln
from scipy.optimize import brentq

# RDP orders to search over when converting to (ε, δ)-DP.
# Low orders are typically tightest for small ε; high orders help for large δ.
_ORDERS = list(range(2, 65)) + [128, 256, 512]


def _log_comb(n: int, k: int) -> float:
    """Log of binomial coefficient C(n, k), computed via log-gamma for stability."""
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


def _rdp_gaussian_subsampled(q: float, sigma: float, alpha: int) -> float:
    """
    RDP of one step of Gaussian mechanism with Poisson subsampling at rate q.

    Computes D_α(M_q(D) ‖ M_q(D')) for neighboring datasets D, D' via:

        (1/(α-1)) · log Σ_{k=0}^{α} C(α,k) · q^k · (1-q)^{α-k} · exp(k(k-1)/(2σ²))

    Args:
        q:     Poisson subsampling rate = batch_size / dataset_size
        sigma: Gaussian noise multiplier
        alpha: RDP order (integer ≥ 2)

    Returns:
        RDP ε at order α (always ≥ 0)
    """
    if q == 0.0:
        return 0.0
    if q == 1.0:
        return float(alpha) / (2.0 * sigma ** 2)

    alpha = int(alpha)
    if alpha < 2:
        return q * float(alpha) / (2.0 * sigma ** 2)

    # Compute log of the moment via logaddexp (numerically stable, no exp() overflow)
    log_sum = -np.inf
    for k in range(alpha + 1):
        log_term = (
            _log_comb(alpha, k)
            + k * np.log(q)
            + (alpha - k) * np.log1p(-q)
            + k * (k - 1) / (2.0 * sigma ** 2)
        )
        log_sum = np.logaddexp(log_sum, log_term)

    return log_sum / (alpha - 1.0)


def _rdp_to_dp(rdp: float, alpha: int, delta: float) -> float:
    """
    Convert an RDP guarantee to an (ε, δ)-DP guarantee.

    Uses the tight conversion from Proposition 3 of Balle et al. (2020),
    "Hypothesis Testing Interpretations and Renyi Differential Privacy".
    """
    return (
        rdp
        + np.log((alpha - 1.0) / alpha)
        - (np.log(delta) + np.log(alpha - 1.0)) / (alpha - 1.0)
    )


def compute_epsilon(
    q: float,
    sigma: float,
    steps: int,
    delta: float,
    orders=None,
) -> float:
    """
    Compute the (ε, δ)-DP guarantee for DP-SGD.

    Args:
        q:      Poisson subsampling rate = batch_size / dataset_size
        sigma:  Gaussian noise multiplier
        steps:  Total number of gradient steps
        delta:  Target δ
        orders: RDP orders to evaluate (default: _ORDERS)

    Returns:
        Minimum ε over all tested orders (tightest bound).
    """
    if orders is None:
        orders = _ORDERS

    best_eps = np.inf
    for alpha in orders:
        rdp = steps * _rdp_gaussian_subsampled(q, sigma, alpha)
        eps = _rdp_to_dp(rdp, alpha, delta)
        if np.isfinite(eps) and eps < best_eps:
            best_eps = eps
    return best_eps


def get_noise_multiplier(
    target_epsilon: float,
    target_delta: float,
    sample_rate: float,
    epochs: int,
    steps: int = None,
) -> float:
    """
    Find the minimum Gaussian noise multiplier σ such that DP-SGD satisfies
    (target_epsilon, target_delta)-DP.

    Uses binary search over σ with RDP accounting (Gaussian mechanism +
    Poisson subsampling). This replaces the opacus `get_noise_multiplier`
    utility with an equivalent self-contained implementation.

    Args:
        target_epsilon: desired ε budget
        target_delta:   desired δ budget
        sample_rate:    Poisson subsampling rate = batch_size / dataset_size
        epochs:         number of training epochs (used if steps is None)
        steps:          total gradient steps (overrides epochs * 1/sample_rate)

    Returns:
        σ: minimum noise multiplier achieving the privacy budget

    Raises:
        ValueError: if target cannot be achieved within the search range
    """
    if steps is None:
        steps = int(np.ceil(epochs / sample_rate))

    def eps_fn(sigma: float) -> float:
        return compute_epsilon(sample_rate, sigma, steps, target_delta)

    sigma_low, sigma_high = 0.01, 1000.0

    eps_at_high = eps_fn(sigma_high)
    if eps_at_high > target_epsilon:
        raise ValueError(
            f"Cannot achieve ε={target_epsilon} even with σ={sigma_high} "
            f"(best ε={eps_at_high:.4f}). Consider increasing target_epsilon or target_delta."
        )

    if eps_fn(sigma_low) <= target_epsilon:
        return sigma_low

    sigma = brentq(
        lambda s: eps_fn(s) - target_epsilon,
        sigma_low,
        sigma_high,
        xtol=1e-6,
    )
    return float(sigma)
