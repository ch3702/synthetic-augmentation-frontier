"""
experiments.py
==============
Train/test split experiments for the synthetic augmentation size-weight frontier.

For each configuration (model, n, alpha, ci_type) we:
  1. Split the questions into train (60%) and test (40%).
  2. On the train set: compute the empirical coverage grid and learn
     the frontier k_hat(lambda).
  3. On the test set: compute the oracle frontier k_hat_oracle(lambda)
     and evaluate the learned frontier's coverage + CI width.
  4. Repeat steps 1-3 with NUM_SPLITS different random splits and
     average the results.

The main entry point is `run_all_experiments(...)`.  
Intermediate results are cached as pickled checkpoints so an interrupted run can be resumed.
"""

from __future__ import annotations

import copy
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from core import (
    alpha_prime as get_alpha_prime,
    combined_CI,
    compute_coverage_grid,
    compute_oracle_frontier,
    learn_frontier,
)



TRAIN_RATIO = 3 / 5         # 60% train, 40% test
WEIGHTS_GRID = np.round(np.arange(0.0, 1.05, 0.05), 2)
GAMMA = 0.5                 # unified proxy constant (see core.proxy_CI)


# ---------------------------------------------------------------------------
# One train/test split
# ---------------------------------------------------------------------------

def one_split_run(
    real_answers: dict[str, np.ndarray],
    syn_answers: dict[str, np.ndarray],
    questions: list[str],
    n: int,
    alpha: float,
    ci_type: str,
    seed: int,
    response_range: float,
    param_range: tuple[float, float],
    weights_grid: np.ndarray = WEIGHTS_GRID,
    k_grid: np.ndarray | None = None,
    n_max: int = 500,
    true_means: dict[str, float] | None = None,
    proxy_method: str = "split",
    gamma: float = GAMMA,
) -> dict:
    """
    Run ONE train/test split.

    The return dict contains all the per-split arrays the caller wants:

      k_hat                 (n_w,)        learned frontier on train
      k_hat_oracle          (n_w,)        oracle  frontier on test
      coverage_train        (n_w, n_k)    p_hat_n on train
      coverage_test_learned (n_w,)        test coverage at k_hat(lambda)
      oracle_cov            (n_w, n_k)    p_n on test
      ci_width              (n_w,)        mean test CI width at k_hat
      ci_width_by_question  (n_w, n_test) per-question test CI width
      eff_size              (n_w,)        lambda * k_hat
      train_questions, test_questions
    """
    if k_grid is None:
        raise ValueError("Please pass a k_grid.")
    if n > n_max:
        raise ValueError(f"n={n} cannot exceed n_max={n_max}.")

    # 1. split questions into train/test
    rng = random.Random(seed)
    shuffled = copy.copy(questions)
    rng.shuffle(shuffled)
    split_idx = int(TRAIN_RATIO * len(shuffled))
    train_questions = shuffled[:split_idx]
    test_questions = shuffled[split_idx:]

    # Per-split response resampling.  The WVS pool is ordered, so we draw a
    # fresh random iid sample of n_max real responses per question using the
    # split seed.  Downstream, the augmented CI uses real_pool[q][:n] and the
    # proxy CI uses real_pool[q][:n_max], both varying across splits -- so the
    # reported coverage/SE average over the sampling of the n real responses,
    # not only over the train/test partition.  Ground-truth means come from the
    # full population (passed in via true_means, computed once).
    resp_rng = np.random.default_rng(seed)
    real_pool = {
        q: resp_rng.choice(real_answers[q], size=n_max, replace=True)
        for q in questions
    }
    if true_means is None:
        true_means = {q: float(np.mean(real_answers[q])) for q in questions}

    # 2. train: empirical coverage grid + frontier
    coverage_train = compute_coverage_grid(
        real_pool, syn_answers, train_questions,
        n, weights_grid, k_grid,
        alpha, response_range, param_range,
        ci_type, n_max, proxy_method, gamma,
    )
    ap = get_alpha_prime(alpha, gamma)
    k_hat = learn_frontier(coverage_train, k_grid, ap, weights_grid)

    # 3. test: oracle frontier
    k_hat_oracle, oracle_cov = compute_oracle_frontier(
        real_pool, syn_answers, test_questions,
        n, weights_grid, k_grid, alpha,
        response_range, param_range, ci_type, n_max,
        true_means=true_means,
    )

    # 4. test: evaluate the learned frontier's coverage and CI width
    n_w = len(weights_grid)
    cov_test_learned = np.zeros(n_w, dtype=float)
    ci_width = np.zeros(n_w, dtype=float)
    ci_width_by_q = np.zeros((n_w, len(test_questions)), dtype=float)

    # Ground-truth means: full-population means (passed in); the augmented CI
    # uses this split's resampled pool.
    real_slices = {q: real_pool[q][:n] for q in test_questions}
    p_lo, p_hi = param_range

    for wi, w in enumerate(weights_grid):
        k = int(k_hat[wi])
        covered = 0
        widths = np.zeros(len(test_questions), dtype=float)
        for qi, q in enumerate(test_questions):
            lo, hi = combined_CI(
                real_slices[q], syn_answers.get(q, np.array([])),
                n, k, float(w), alpha,
                response_range, param_range, ci_type,
            )
            if lo <= true_means[q] <= hi:
                covered += 1
            # clip the width to the parameter range
            widths[qi] = min(hi, p_hi) - max(lo, p_lo)
        cov_test_learned[wi] = covered / len(test_questions)
        ci_width[wi] = float(np.mean(widths))
        ci_width_by_q[wi] = widths

    eff_size = weights_grid * k_hat.astype(float)

    return {
        "k_hat":                 k_hat,
        "k_hat_oracle":          k_hat_oracle,
        "coverage_train":        coverage_train,
        "coverage_test_learned": cov_test_learned,
        "oracle_cov":            oracle_cov,
        "ci_width":              ci_width,
        "ci_width_by_question":  ci_width_by_q,
        "eff_size":              eff_size,
        "train_questions":       train_questions,
        "test_questions":        test_questions,
    }


# ---------------------------------------------------------------------------
# Many train/test splits, aggregated
# ---------------------------------------------------------------------------

def _resume_compatible(
    cached: dict, seeds: list, weights_grid: np.ndarray, k_grid: np.ndarray,
    n_max: int, ci_type: str, n: int, alpha: float,
    proxy_method: str, gamma: float, n_w: int, n_test: int,
) -> bool:
    """
    True if `cached` was produced by the same configuration and can therefore
    have its split rows reused.  Requires a stored seed list whose prefix
    matches the current seed stream, plus matching grids and hyperparameters.
    """
    try:
        c_seeds = cached.get("seeds")
        if c_seeds is None:
            return False                       # pre-resume checkpoint: recompute
        m = min(int(cached["k_hats"].shape[0]), len(seeds))
        return (
            list(c_seeds)[:m] == list(seeds)[:m]
            and np.array_equal(cached.get("weights_grid"), weights_grid)
            and np.array_equal(cached.get("k_grid"), k_grid)
            and cached.get("n_max") == n_max
            and cached.get("ci_type") == ci_type
            and cached.get("n") == n
            and cached.get("alpha") == alpha
            and cached.get("proxy_method", "full") == proxy_method
            and cached.get("gamma", 0.5) == gamma
            and cached["k_hats"].shape[1] == n_w
            and cached["ci_widths_by_question"].shape[2] == n_test
        )
    except (KeyError, AttributeError, IndexError, TypeError):
        return False


def multiple_split_run(
    real_answers: dict[str, np.ndarray],
    syn_answers: dict[str, np.ndarray],
    questions: list[str],
    n: int,
    alpha: float,
    ci_type: str,
    response_range: float,
    param_range: tuple[float, float],
    num_splits: int,
    weights_grid: np.ndarray,
    k_grid: np.ndarray,
    n_max: int = 500,
    master_seed: int = 42,
    proxy_method: str = "split",
    gamma: float = GAMMA,
    cached: dict | None = None,
) -> dict:
    """
    Run `num_splits` train/test splits and aggregate the results.

    The returned dict has per-split arrays of shape (num_splits, n_w),
    plus mean and standard-error summaries across splits.

    Incremental resume: if `cached` is a previous result of this same
    configuration, its already-computed split rows are reused and only the
    remaining seeds are run.  This is safe because the seed stream is a
    prefix-stable function of `master_seed` (drawing `num_splits` seeds shares
    its first `m` values with any earlier run that drew `m <= num_splits`), and
    we additionally verify the stored seeds prefix matches.
    """
    seeds = np.random.default_rng(master_seed).integers(0, 100_000, size=num_splits).tolist()

    # Ground-truth means from the full population, computed once and shared by
    # every split (the per-split resampling only affects the augmented/proxy
    # samples, not the target theta_j).
    true_means = {q: float(np.mean(real_answers[q])) for q in questions}

    n_w = len(weights_grid)
    n_test = len(questions) - int(TRAIN_RATIO * len(questions))

    # per-split storage
    k_hats           = np.zeros((num_splits, n_w), dtype=int)
    k_hat_oracles    = np.zeros((num_splits, n_w), dtype=int)
    cov_test_learned = np.zeros((num_splits, n_w), dtype=float)
    ci_widths        = np.zeros((num_splits, n_w), dtype=float)
    ci_widths_by_q   = np.zeros((num_splits, n_w, n_test), dtype=float)
    eff_sizes        = np.zeros((num_splits, n_w), dtype=float)

    first_split: dict | None = None

    # ---- decide how many split rows can be reused from `cached` ----
    n_old = 0
    if cached is not None and _resume_compatible(
        cached, seeds, weights_grid, k_grid, n_max,
        ci_type, n, alpha, proxy_method, gamma, n_w, n_test,
    ):
        n_old = min(int(cached["k_hats"].shape[0]), num_splits)
        k_hats[:n_old]           = cached["k_hats"][:n_old]
        k_hat_oracles[:n_old]    = cached["k_hat_oracles"][:n_old]
        cov_test_learned[:n_old] = cached["cov_test_learned"][:n_old]
        ci_widths[:n_old]        = cached["ci_widths"][:n_old]
        ci_widths_by_q[:n_old]   = cached["ci_widths_by_question"][:n_old]
        eff_sizes[:n_old]        = cached["eff_sizes"][:n_old]
        first_split = cached.get("first_split")

    if n_old:
        print(f"  n={n}, alpha={alpha}, {ci_type}, {proxy_method}: "
              f"reusing {n_old} split(s), computing {num_splits - n_old} more")

    for i in tqdm(range(n_old, num_splits),
                  desc=f"  n={n}, alpha={alpha}, {ci_type}", initial=n_old, total=num_splits):
        res = one_split_run(
            real_answers, syn_answers, questions,
            n, alpha, ci_type, seeds[i],
            response_range, param_range,
            weights_grid, k_grid, n_max,
            true_means=true_means,
            proxy_method=proxy_method, gamma=gamma,
        )
        k_hats[i]           = res["k_hat"]
        k_hat_oracles[i]    = res["k_hat_oracle"]
        cov_test_learned[i] = res["coverage_test_learned"]
        ci_widths[i]        = res["ci_width"]
        ci_widths_by_q[i]   = res["ci_width_by_question"]
        eff_sizes[i]        = res["eff_size"]
        if i == 0:
            first_split = res

    def _se(arr: np.ndarray) -> np.ndarray:
        return arr.std(axis=0) / np.sqrt(num_splits)

    return {
        # resume bookkeeping
        "seeds":        seeds,
        "master_seed":  master_seed,
        "_n_reused":    n_old,
        # raw per-split arrays
        "k_hats":                k_hats,
        "k_hat_oracles":         k_hat_oracles,
        "cov_test_learned":      cov_test_learned,
        "ci_widths":             ci_widths,
        "ci_widths_by_question": ci_widths_by_q,
        "eff_sizes":             eff_sizes,
        # mean + SE summaries
        "k_hat_mean":         k_hats.mean(axis=0),
        "k_hat_se":           _se(k_hats),
        "k_hat_oracle_mean":  k_hat_oracles.mean(axis=0),
        "k_hat_oracle_se":    _se(k_hat_oracles),
        "cov_mean":           cov_test_learned.mean(axis=0),
        "cov_se":             _se(cov_test_learned),
        "ci_width_mean":      ci_widths.mean(axis=0),
        "ci_width_se":        _se(ci_widths),
        "eff_size_mean":      eff_sizes.mean(axis=0),
        "eff_size_se":        _se(eff_sizes),
        # one reference split, used for the frontier-dominance plot
        "first_split":        first_split,
        # metadata
        "weights_grid": weights_grid,
        "k_grid":       k_grid,
        "n": n, "alpha": alpha, "n_max": n_max,
        "ci_type": ci_type,
        "proxy_method": proxy_method, "gamma": gamma,
    }


# ---------------------------------------------------------------------------
# Top-level runner with checkpointing
# ---------------------------------------------------------------------------

def _checkpoint_path(
    checkpoint_dir: Path, model: str, n: int, alpha: float,
    ct: str, n_max: int,
    proxy_method: str = "split", gamma: float = GAMMA,
) -> Path:
    # The unified proxy constant gamma is encoded once; the proxy method adds a
    # trailing "_full" / "_split" tag so the two schemes don't collide.
    proxy_tag = f"_{proxy_method}"
    return checkpoint_dir / (
        f"{model}_n{n}_alpha{int(alpha*100):02d}_{ct}"
        f"_gamma{int(gamma*100):02d}_nmax{int(n_max)}{proxy_tag}.pkl"
    )


def run_all_experiments(
    real_answers: dict[str, np.ndarray],
    all_syn_answers: dict[str, dict[str, np.ndarray]],
    output_dir: str | Path,
    models: list[str],
    n_values: list[int],
    alpha_values: list[float],
    ci_types: list[str],
    response_range: float,
    param_range: tuple[float, float],
    num_splits: int,
    weights_grid: np.ndarray,
    k_grid: np.ndarray,
    n_max: int = 500,
    master_seed: int = 42,
    proxy_method: str = "split",
    gamma: float = GAMMA,
) -> dict:
    """
    Loop over every (model, n, alpha, ci_type) combination, run multiple
    splits, and save a checkpoint pickle per combination.

    Returns a nested dict:
        results[model][n][alpha][ci_type] = multiple_split_run output

    Also writes one summary CSV per ci_type into output_dir.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    questions = list(real_answers.keys())
    print(f"  {len(questions)} questions, {len(models)} models")

    # Build the full configuration grid up-front.
    configs = [
        (model, n, alpha, ct)
        for model in models
        for n in n_values
        for alpha in alpha_values
        for ct in ci_types
    ]

    results: dict = {}
    for model, n, alpha, ct in tqdm(configs, desc="Configurations", unit="config"):
        results.setdefault(model, {}).setdefault(n, {}).setdefault(alpha, {})

        ckpt = _checkpoint_path(checkpoint_dir, model, n, alpha, ct, n_max,
                                proxy_method, gamma)
        cached = None
        if ckpt.exists():
            with open(ckpt, "rb") as f:
                cached = pickle.load(f)

        # multiple_split_run reuses compatible cached split rows and computes
        # only the additional seeds needed to reach num_splits.
        res = multiple_split_run(
            real_answers, all_syn_answers[model], questions,
            n, alpha, ct,
            response_range, param_range,
            num_splits, weights_grid, k_grid,
            n_max, master_seed,
            proxy_method, gamma,
            cached=cached,
        )

        # Save only when we actually computed new split rows.  If everything
        # was reused (num_splits <= cached splits), leave the on-disk checkpoint
        # untouched so a smaller NUM_SPLITS never discards existing work.
        if res.get("_n_reused", 0) < num_splits:
            with open(ckpt, "wb") as f:
                pickle.dump(res, f)
        results[model][n][alpha][ct] = res

    _save_summary_csvs(results, output_dir, models, n_values, alpha_values, ci_types)
    return results


# ---------------------------------------------------------------------------
# Summary CSV writer
# ---------------------------------------------------------------------------

def _save_summary_csvs(
    results: dict,
    output_dir: Path,
    models: list[str],
    n_values: list[int],
    alpha_values: list[float],
    ci_types: list[str],
) -> None:
    """One summary CSV per ci_type."""
    for ct in ci_types:
        rows = []
        for model in models:
            if model not in results:
                continue
            for n in n_values:
                for alpha in alpha_values:
                    res = results[model][n][alpha][ct]
                    wg = res["weights_grid"]
                    for wi, w in enumerate(wg):
                        rows.append({
                            "model":             model,
                            "n":                 n,
                            "alpha":             alpha,
                            "weight":            w,
                            "k_hat_mean":        res["k_hat_mean"][wi],
                            "k_hat_se":          res["k_hat_se"][wi],
                            "k_hat_oracle_mean": res["k_hat_oracle_mean"][wi],
                            "k_hat_oracle_se":   res["k_hat_oracle_se"][wi],
                            "coverage_mean":     res["cov_mean"][wi],
                            "coverage_se":       res["cov_se"][wi],
                            "ci_width_mean":     res["ci_width_mean"][wi],
                            "ci_width_se":       res["ci_width_se"][wi],
                            "eff_size_mean":     res["eff_size_mean"][wi],
                            "eff_size_se":       res["eff_size_se"][wi],
                        })
        fpath = output_dir / f"summary_{ct}.csv"
        pd.DataFrame(rows).to_csv(fpath, index=False)
        print(f"  Saved {fpath.name}")
