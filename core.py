"""
core.py
=======
Mathematical building blocks for the synthetic augmentation size-weight frontier.

This file contains the four key ideas of the paper:

  1. combined_CI(...) — confidence interval for the mean using
     n real samples plus k synthetic samples weighted by lambda.
  2. proxy_CI(...)    — a stand-in for the unknown true mean,
     computed from real samples.
  3. compute_coverage_grid(...) — empirical coverage probability
     p_hat_n(lambda, k) on a grid of (lambda, k) values.
  4. learn_frontier(...) — non-increasing curve k_hat(lambda)
     giving, for each lambda, the largest k that still meets
     a target coverage level.

There is also compute_oracle_frontier(...) which is the "ground truth"
analogue of (3)+(4) used to evaluate how good the learned frontier is.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.stats import norm
from tqdm.auto import tqdm


@lru_cache(maxsize=None)
def _z(alpha: float) -> float:
    """Cached two-sided normal quantile z_{1-alpha/2}.
    """
    return float(norm.ppf(1.0 - alpha / 2.0))


# ---------------------------------------------------------------------------
# Plot styling: maps model name -> {label, color, marker}.
# Used by plots.py.
# ---------------------------------------------------------------------------

LLM_PLOT_INFO = {
    # WorldValuesBench simulators.
    "gpt4o":       {"label": "GPT-4o",       "color": "#1f77b4", "marker": "o"},
    "gpt-5-mini":  {"label": "GPT-5 mini",   "color": "#ff7f0e", "marker": "s"},
    "llama-3.3":   {"label": "Llama 3.3 70B", "color": "#9467bd", "marker": "^"},
    "Qwen3-235B":  {"label": "Qwen3-235B",   "color": "#2ca02c", "marker": "D"},
    "uniform":     {"label": "Uniform",      "color": "#7f7f7f", "marker": "x"},
}


# ---------------------------------------------------------------------------
# Combined confidence intervals (real + synthetic, weight lambda)
# ---------------------------------------------------------------------------

def combined_CI(
    real_samples: np.ndarray,
    syn_samples: np.ndarray,
    n: int,
    k: int,
    weight: float,
    alpha: float,
    response_range: float,
    param_range: tuple[float, float],
    ci_type: str = "bernstein",
) -> tuple[float, float]:
    """
    Confidence interval for the mean using n real + k synthetic samples.

    Each synthetic sample is treated as "weight" copies of a real sample.

    Parameters
    ----------
    real_samples   : 1-D array, at least `n` entries
    syn_samples    : 1-D array, at least `k` entries (may be empty if k=0)
    n              : number of real samples used (uses real_samples[:n])
    k              : number of synthetic samples used (uses syn_samples[:k])
    weight         : lambda in [0, 1]
    alpha          : target miscoverage level
    response_range : range M of the responses (max - min); used by Hoeffding
    param_range    : (lo, hi) of the parameter space; used as a fallback
                     when not enough data is available
    ci_type        : "hoeffding", "clt", or "bernstein"

    Returns
    -------
    (lower, upper) — the CI endpoints
    """
    ci_type = ci_type.lower()
    p_lo, p_hi = param_range

    # Fall back to the full parameter range if we have too little data.
    if n < 2:
        return (p_lo, p_hi)

    x = real_samples[:n]
    mean_x = float(np.mean(x))
    s_x = float(np.std(x, ddof=1))

    # --- Pure real-only case (k=0 or lambda=0) -----------------------------
    if k <= 0 or weight == 0.0:
        if ci_type == "clt":
            if n <= 5:
                return (p_lo, p_hi)
            z = _z(alpha)
            me = z * s_x / np.sqrt(n)
        elif ci_type == "hoeffding":
            me = response_range * np.sqrt(np.log(2.0 / alpha) / (2.0 * n))
        elif ci_type == "bernstein":
            # Empirical Bernstein bound (Maurer & Pontil, 2009), M = response_range.
            log_term = np.log(4.0 / alpha)
            me = (s_x * np.sqrt(2.0 * log_term / n)
                  + 7.0 * response_range * log_term / (3.0 * max(n - 1, 1)))
        else:
            raise ValueError(f"Unknown ci_type: {ci_type!r}.")
        return (mean_x - me, mean_x + me)

    # --- Combined real + synthetic case ------------------------------------
    k_use = min(k, len(syn_samples))
    if k_use < 2:
        # Not enough synthetic data; fall back to real-only.
        return combined_CI(real_samples, syn_samples, n, 0, weight, alpha,
                           response_range, param_range, ci_type)

    y = syn_samples[:k_use]
    mean_y = float(np.mean(y))
    s_y = float(np.std(y, ddof=1))

    n_eff = n + weight * k_use                                # effective sample size
    theta_hat = (np.sum(x) + weight * np.sum(y)) / n_eff      # weighted mean estimator
    var = (n * s_x ** 2 + k_use * weight ** 2 * s_y ** 2) / (n_eff ** 2)
    sigma = np.sqrt(max(var, 0.0))                            # plug-in SE of theta_hat

    if ci_type == "clt":
        if n + k_use <= 5:
            return (p_lo, p_hi)
        z = _z(alpha)
        me = z * sigma
    elif ci_type == "hoeffding":
        # Weighted Hoeffding: variance bound = M^2 * (n + lambda^2*k) / n_eff^2.
        me = response_range * np.sqrt(
            np.log(2.0 / alpha) * (n + weight ** 2 * k_use) / 2.0
        ) / n_eff
    elif ci_type == "bernstein":
        # Augmented empirical Bernstein bound (paper eq. margin-Bern-aug):
        #   r = sqrt(2 * v_hat * log(4/alpha)) + 7 M log(4/alpha) / (3 (n_eff - 1)),
        # where v_hat = sigma^2 is the plug-in variance of theta_hat and M is the
        # response range.  `sigma` already includes the 1/n_eff shrinkage, so the
        # first term is sqrt(2 log(4/alpha)) * sigma.  Reduces to the real-only
        # empirical Bernstein bound when k=0 or weight=0.
        log_term = np.log(4.0 / alpha)
        me = (sigma * np.sqrt(2.0 * log_term)
              + 7.0 * response_range * log_term / (3.0 * max(n_eff - 1.0, 1.0)))
    else:
        raise ValueError(f"Unknown ci_type: {ci_type!r}.")

    return (theta_hat - me, theta_hat + me)


# ---------------------------------------------------------------------------
# Proxy CI for the unknown true mean
# ---------------------------------------------------------------------------

def proxy_CI(
    real_all: np.ndarray,
    alpha: float,
    param_range: tuple[float, float],
    n_max: int,
    method: str = "full",
    gamma: float = 0.5,
    n_ci: int = 0,
) -> tuple[float, float]:
    """
    Build a proxy interval for the unknown true mean theta_j.

    Both schemes share a single constant gamma in (0, 1) (see paper, Lemma 1):

    full  (method="full"):
        Uses the full real-sample budget real_all[:n_max] at the higher
        coverage level 1 - gamma*alpha.  These n_max samples overlap the ones
        used inside the combined CI (allowed: same question).  Paired with
        alpha' = (1 - gamma) * alpha.

    split (method="split"):
        Uses the DISJOINT tail real_all[n_ci:n_max] (the n_max - n_ci samples
        NOT used by the combined CI) at coverage level 1 - gamma.  The proxy is
        then independent of the combined CI.  Paired with alpha' = (1 - gamma) * alpha.
    """
    p_lo, p_hi = param_range
    if method == "full":
        data = real_all[:n_max]
        coverage = 1.0 - gamma * alpha
    elif method == "split":
        data = real_all[n_ci:n_max]
        coverage = 1.0 - gamma
    else:
        raise ValueError(f"Unknown proxy method: {method!r}. Choose 'full' or 'split'.")

    n = len(data)
    if n <= 5:
        return (p_lo, p_hi)

    m = float(np.mean(data))
    s = float(np.std(data, ddof=1))
    z = norm.ppf(0.5 + coverage / 2.0)
    me = z * s / np.sqrt(n)
    return (m - me, m + me)


def alpha_prime(alpha: float, gamma: float = 0.5) -> float:
    """
    Target miscoverage level used to learn the frontier.
    Both proxy schemes share the same acceptance threshold
    tau = 1 - (1 - gamma) * alpha, that is,

        alpha' = (1 - gamma) * alpha
    """
    return (1.0 - gamma) * alpha


# ---------------------------------------------------------------------------
# Empirical coverage grid p_hat_n(lambda, k)
# ---------------------------------------------------------------------------

def compute_coverage_grid(
    real_answers: dict[str, np.ndarray],
    syn_answers: dict[str, np.ndarray],
    questions: list[str],
    n: int,
    weights_grid: np.ndarray,
    k_grid: np.ndarray,
    alpha: float,
    response_range: float,
    param_range: tuple[float, float],
    ci_type: str = "bernstein",
    n_max: int = 500,
    proxy_method: str = "split",
    gamma: float = 0.5,
) -> np.ndarray:
    """
    For each (lambda, k), count the fraction of training questions where
    the proxy CI is contained inside the combined CI:

        proxy_CI(q)  subseteq  combined_CI_n(q ; lambda, k)

    Returns a 2-D array of shape (len(weights_grid), len(k_grid)).
    """
    if n > n_max:
        raise ValueError(f"n={n} cannot exceed n_max={n_max}.")

    # Precompute proxy CIs and the data slices used in the combined CI.
    proxy_intervals: dict[str, tuple[float, float]] = {}
    real_slices: dict[str, np.ndarray] = {}
    syn_slices: dict[str, np.ndarray] = {}
    for q in questions:
        real_all = real_answers[q]
        proxy_intervals[q] = proxy_CI(real_all, alpha, param_range, n_max=n_max,
                                      method=proxy_method, gamma=gamma, n_ci=n)
        real_slices[q] = real_all[:n]
        syn_slices[q] = syn_answers.get(q, np.array([]))

    n_w, n_k = len(weights_grid), len(k_grid)
    coverage = np.zeros((n_w, n_k), dtype=float)

    total = n_w * n_k
    with tqdm(total=total, desc="coverage grid", unit="cell", leave=False) as pbar:
        for wi, w in enumerate(weights_grid):
            for ki, k in enumerate(k_grid):
                covered = 0
                for q in questions:
                    lo_p, hi_p = proxy_intervals[q]
                    lo_c, hi_c = combined_CI(
                        real_slices[q], syn_slices[q], n, int(k), float(w), alpha,
                        response_range, param_range, ci_type,
                    )
                    # Proxy CI contained in combined CI
                    if lo_c <= lo_p and hi_c >= hi_p:
                        covered += 1
                coverage[wi, ki] = covered / len(questions)
                pbar.update(1)

    return coverage


# ---------------------------------------------------------------------------
# Frontier learning: k_hat(lambda)
# ---------------------------------------------------------------------------

def learn_frontier(
    coverage_grid: np.ndarray,
    k_grid: np.ndarray,
    alpha_prime_value: float,
    weights_grid: np.ndarray | None = None,
) -> np.ndarray:
    """
    Learn the non-increasing empirical frontier k_hat(lambda).

    For each lambda index wi:
      k_pre[wi] = largest k in k_grid such that coverage_grid[wi, :]
                   stays >= 1 - alpha_prime for all k' <= k.

    Special case: at lambda=0 the synthetic data has zero weight and
    cannot affect the CI, so k_pre[wi] is set to the largest k available.

    Finally, monotonicity is enforced (a running min from left to right)
    because the frontier must be non-increasing in lambda.
    """
    n_w, _ = coverage_grid.shape
    threshold = 1.0 - alpha_prime_value
    k_pre = np.zeros(n_w, dtype=int)

    for wi in range(n_w):
        if weights_grid is not None and weights_grid[wi] == 0.0:
            k_pre[wi] = int(k_grid[-1])
            continue

        row = coverage_grid[wi]
        fail_idx = np.where(row < threshold)[0]
        if len(fail_idx) == 0:
            k_pre[wi] = int(k_grid[-1])     # never fails -> use the max k
        elif fail_idx[0] == 0:
            k_pre[wi] = 0                    # fails even at the smallest k
        else:
            k_pre[wi] = int(k_grid[fail_idx[0] - 1])

    # Enforce non-increasing in lambda (lower isotonic envelope).
    return np.minimum.accumulate(k_pre)


# ---------------------------------------------------------------------------
# Oracle frontier (uses ground-truth means computed from all real samples)
# ---------------------------------------------------------------------------

def compute_oracle_frontier(
    real_answers: dict[str, np.ndarray],
    syn_answers: dict[str, np.ndarray],
    test_questions: list[str],
    n: int,
    weights_grid: np.ndarray,
    k_grid: np.ndarray,
    alpha: float,
    response_range: float,
    param_range: tuple[float, float],
    ci_type: str = "hoeffding",
    n_max: int = 500,
    true_means: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the *oracle* frontier on the test set.

    We treat the mean of ALL real responses as a proxy for the true mean
    theta_j (valid because the datasets have many real responses).
    Then we count how often the combined CI covers that true mean, for
    every (lambda, k), and apply the same non-increasing rule used by
    learn_frontier.

    ``real_answers`` here holds the per-split *resampled* pools (one random
    iid draw of real responses per question); the ground-truth means come
    from the full population and are passed in via ``true_means``.  When
    ``true_means`` is None (e.g. called standalone on full arrays), it falls
    back to the held-out tail mean.

    Returns
    -------
    k_hat_oracle : 1-D int array of length len(weights_grid)
    oracle_cov   : 2-D float array of shape (len(weights_grid), len(k_grid))
    """
    if n > n_max:
        raise ValueError(f"n={n} cannot exceed n_max={n_max}.")

    if true_means is None:
        true_means = {q: float(np.mean(real_answers[q][n_max:])) for q in test_questions}
    real_slices = {q: real_answers[q][:n] for q in test_questions}
    syn_slices = {q: syn_answers.get(q, np.array([])) for q in test_questions}

    n_w, n_k = len(weights_grid), len(k_grid)
    oracle_cov = np.zeros((n_w, n_k), dtype=float)

    total = n_w * n_k
    with tqdm(total=total, desc="oracle grid", unit="cell", leave=False) as pbar:
        for wi, w in enumerate(weights_grid):
            for ki, k in enumerate(k_grid):
                covered = 0
                for q in test_questions:
                    lo, hi = combined_CI(
                        real_slices[q], syn_slices[q], n, int(k), float(w), alpha,
                        response_range, param_range, ci_type,
                    )
                    if lo <= true_means[q] <= hi:
                        covered += 1
                oracle_cov[wi, ki] = covered / len(test_questions)
                pbar.update(1)

    k_hat_oracle = learn_frontier(oracle_cov, k_grid, alpha, weights_grid)
    return k_hat_oracle, oracle_cov
