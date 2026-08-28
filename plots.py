"""
plots.py
========
Main-text figures for WorldValuesBench, empirical Bernstein CI, alpha = 0.1.

Reads the 50-split checkpoints written by run_worldvalue.py from
    results/worldvalue_full50/checkpoints/   (full proxy)
    results/worldvalue_split50/checkpoints/  (split proxy)
and writes the PDFs to results/figures/.

Produces five figures, all with three horizontally aligned panels (n = 10, 30, 50):

  Proxy comparison (both proxies overlaid; one curve per model):
    plot_coverage.pdf         -- test coverage at the learned frontier
    plot_width_reduction.pdf  -- relative CI-width reduction R_n(lambda)
    plot_effective_size.pdf   -- effective synthetic size lambda * k_hat_n(lambda)

  Learned vs oracle frontier (GPT-4o, one seed = first split; one file per proxy):
    plot_frontier_full.pdf    -- full-proxy learned frontier vs oracle
    plot_frontier_split.pdf   -- split-proxy learned frontier vs oracle

Encoding (proxy-comparison figures)
    model  -> color + marker   (GPT-4o = blue circles, GPT-5 mini = orange squares)
    proxy  -> line style + fill (split = solid/filled, full = dashed/open)

Run from this directory:

    python plots.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mplconfig"))

import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter

from core import LLM_PLOT_INFO


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RESULTS_DIR = HERE / "results"
FULL_DIR    = RESULTS_DIR / "worldvalue_full50"  / "checkpoints"
SPLIT_DIR   = RESULTS_DIR / "worldvalue_split50" / "checkpoints"
OUT_DIR     = RESULTS_DIR / "figures"

ALPHA   = 0.10
CI_TYPE = "bernstein"
N_MAX   = 500
GAMMA   = 0.5           # unified proxy constant (see core.proxy_CI)

MODELS  = ["gpt4o", "gpt-5-mini"]     # color + marker order
N_PANELS = [10, 30, 50]               # one panel per n

FRONTIER_MODEL = "gpt4o"              # model shown in the frontier figures
ORACLE_COLOR   = "#d62728"            # oracle frontier in red (learned uses the model color)
ORACLE_MARKER  = "x"                  # oracle node marker (learned uses the model marker)

# proxy -> line style / marker fill
PROXY_STYLE = {
    "split": {"ls": "-",  "fill": True,  "label": "Split proxy"},
    "full":  {"ls": "--", "fill": False, "label": "Full proxy"},
}

# shared 3-panel layout
FIGSIZE = (11.5, 4.2)
SUBPLOTS_ADJUST = dict(left=0.065, right=0.99, top=0.90, bottom=0.26, wspace=0.16)


# ---------------------------------------------------------------------------
# Data loading / metric extraction
# ---------------------------------------------------------------------------

def _ckpt_path(proxy: str, model: str, n: int) -> Path:
    base = (f"{model}_n{n}_alpha{int(ALPHA*100):02d}_{CI_TYPE}"
            f"_gamma{int(GAMMA*100):02d}_nmax{int(N_MAX)}")
    if proxy == "split":
        return SPLIT_DIR / f"{base}_split.pkl"
    return FULL_DIR / f"{base}_full.pkl"


def _load(proxy: str, model: str, n: int) -> dict:
    path = _ckpt_path(proxy, model, n)
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def _width_reduction_stats(res: dict) -> tuple[np.ndarray, np.ndarray]:
    """Relative width reduction R_n(lambda) = mean_q[1 - |CI(lambda,k_hat)|/|CI(0,0)|].

    Per-split average over test questions (baseline = lambda=0 width), then
    mean +/- SE across the 50 splits.
    """
    widths = np.asarray(res["ci_widths_by_question"], dtype=float)   # (splits, n_w, n_test)
    baseline = widths[:, [0], :]
    reductions = 1.0 - np.divide(
        widths, baseline,
        out=np.full_like(widths, np.nan, dtype=float),
        where=baseline != 0,
    )
    split_means = np.nanmean(reductions, axis=2)                     # (splits, n_w)
    counts = np.sum(np.isfinite(split_means), axis=0)
    mean = np.nanmean(split_means, axis=0)
    se = np.nanstd(split_means, axis=0) / np.sqrt(np.maximum(counts, 1))
    se[counts == 0] = np.nan
    return mean, se


def _metric(res: dict, which: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean, se) arrays over the weight grid for the requested metric."""
    if which == "coverage":
        return res["cov_mean"], res["cov_se"]
    if which == "width_reduction":
        return _width_reduction_stats(res)
    if which == "effective_size":
        return res["eff_size_mean"], res["eff_size_se"]
    raise ValueError(which)


# ---------------------------------------------------------------------------
# K-limited region: largest weight up to which some displayed frontier is
# still capped at k_hat = K (= k_grid[-1]).  Determined from the data.
# ---------------------------------------------------------------------------

def _k_limited_edge(panel_results: list[dict], weights: np.ndarray) -> float:
    """Right edge (in lambda) of the region where AT LEAST ONE per-seed frontier
    is still capped at k_hat == K.

    Uses the raw per-split frontiers (res["k_hats"], shape (n_splits, n_w)).  For
    every seed of every displayed curve we find the first weight at which that
    seed drops below K; the region extends to the LAST such departure (the max
    over all seeds and curves), i.e. up to the largest lambda at which some seed
    is still pinned at the budget.
    """
    n_w = len(weights)
    K = int(panel_results[0]["k_grid"][-1])
    last_drop = 0                                 # max over seeds/curves of first-drop index
    for res in panel_results:
        kh = np.asarray(res["k_hats"], dtype=float)      # (n_splits, n_w)
        below = kh < K - 1e-9
        has_drop = below.any(axis=1)
        first = np.where(has_drop, below.argmax(axis=1), n_w)   # n_w if seed never drops
        last_drop = max(last_drop, int(first.max()))
    if last_drop <= 0:
        return 0.0
    last_k = last_drop - 1                         # last index where some seed still capped
    step = float(weights[1] - weights[0]) if n_w > 1 else 0.05
    if last_k >= n_w - 1:
        return float(weights[-1])
    return float(weights[last_k]) + 0.5 * step     # boundary between capped / uncapped


# ---------------------------------------------------------------------------
# Proxy-comparison figures (coverage / width reduction / effective size)
# ---------------------------------------------------------------------------

METRIC_SPEC = {
    "coverage": {
        "ylabel": "Test coverage",
        "ylim":   (0.87, 1.005),
        "hline":  1.0 - ALPHA,                    # nominal 0.90
        "fname":  "plot_coverage.pdf",
        "percent": False,
    },
    "width_reduction": {
        "ylabel": "Relative width reduction",
        "ylim":   (-0.02, 0.57),
        "hline":  0.0,
        "fname":  "plot_width_reduction.pdf",
        "percent": True,
    },
    "effective_size": {
        "ylabel": "Effective synthetic sample size",
        "ylim":   (0.0, 48.0),
        "hline":  None,
        "fname":  "plot_effective_size.pdf",
        "percent": False,
    },
}


def _make_metric_figure(which: str, data: dict) -> Path:
    spec = METRIC_SPEC[which]
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE, sharex=True, sharey=True)

    for pi, (ax, n) in enumerate(zip(axes, N_PANELS)):
        weights = data[("full", MODELS[0], n)]["weights_grid"]

        # K-limited shading (endpoint from the data)
        panel_res = [data[(px, m, n)] for px in ("split", "full") for m in MODELS]
        edge = _k_limited_edge(panel_res, weights)
        if edge > 0:
            ax.axvspan(weights[0], edge, color="0.85", alpha=0.55, lw=0, zorder=0)

        # reference line
        if spec["hline"] is not None:
            ax.axhline(spec["hline"], color="0.35", ls=":", lw=0.9, zorder=1)

        # curves: model (color/marker) x proxy (style/fill).  Straight segments
        # with a marker at every point, plus a +/- 1.96*SE shaded band.
        for model in MODELS:
            info = LLM_PLOT_INFO[model]
            color, marker = info["color"], info["marker"]
            for proxy in ("split", "full"):
                ps = PROXY_STYLE[proxy]
                mean, se = _metric(data[(proxy, model, n)], which)
                mean = np.asarray(mean, dtype=float)
                band = 1.96 * np.asarray(se, dtype=float)
                ax.fill_between(weights, mean - band, mean + band,
                                color=color, alpha=0.13, lw=0, zorder=1.5)
                ax.plot(weights, mean, ls=ps["ls"], color=color, lw=1.6,
                        marker=marker, ms=5.0, markevery=1,
                        mfc=(color if ps["fill"] else "white"), mec=color,
                        markeredgewidth=1.1, zorder=3)

        ax.set_title(f"$n={n}$", fontsize=13)
        ax.set_xlim(weights[0], weights[-1])
        ax.set_ylim(*spec["ylim"])
        ax.tick_params(labelsize=10.5, labelleft=True)   # y-tick labels on every panel
        ax.set_xlabel(r"Synthetic weight $\lambda$", fontsize=11)
        if spec["percent"]:
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        if pi == 0:
            ax.set_ylabel(spec["ylabel"], fontsize=12)

    # one shared legend below the panels: models + proxies + K-limited patch
    model_handles = [
        Line2D([0], [0], color=LLM_PLOT_INFO[m]["color"], marker=LLM_PLOT_INFO[m]["marker"],
               ls="-", lw=1.6, mfc=LLM_PLOT_INFO[m]["color"], mec=LLM_PLOT_INFO[m]["color"],
               ms=5.5, label=LLM_PLOT_INFO[m]["label"])
        for m in MODELS
    ]
    proxy_handles = [
        Line2D([0], [0], color="0.25", marker="o", ls=PROXY_STYLE[p]["ls"], lw=1.6,
               mfc=("0.25" if PROXY_STYLE[p]["fill"] else "white"), mec="0.25",
               ms=5.5, label=PROXY_STYLE[p]["label"])
        for p in ("split", "full")
    ]
    region_handle = [Patch(facecolor="0.85", alpha=0.55, label="$K$-limited")]
    handles = model_handles + proxy_handles + region_handle
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               fontsize=11, frameon=False, bbox_to_anchor=(0.5, 0.02),
               handletextpad=0.5, columnspacing=1.8)

    fig.subplots_adjust(**SUBPLOTS_ADJUST)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / spec["fname"]
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Learned vs oracle frontier figure (GPT-4o, one seed = first split)
# ---------------------------------------------------------------------------

def _make_frontier_figure(proxy: str, data: dict) -> Path:
    """One seed's learned frontier k_hat(lambda) (model color) vs the oracle
    frontier k_hat*(lambda) (red), both solid step curves, for FRONTIER_MODEL,
    one panel per n."""
    info = LLM_PLOT_INFO[FRONTIER_MODEL]
    color = info["color"]
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE, sharex=True, sharey=True)

    for pi, (ax, n) in enumerate(zip(axes, N_PANELS)):
        res = data[(proxy, FRONTIER_MODEL, n)]
        weights = np.asarray(res["weights_grid"], dtype=float)
        first = res["first_split"]
        learned = np.asarray(first["k_hat"], dtype=float)
        oracle = np.asarray(first["k_hat_oracle"], dtype=float)
        K = int(res["k_grid"][-1])

        # step functions (non-increasing frontiers) with a marker at each grid
        # node, so the true per-lambda values are visible; oracle under, learned on top
        ax.step(weights, oracle, where="post", color=ORACLE_COLOR, ls="-", lw=1.0,
                marker=ORACLE_MARKER, ms=3.0, markevery=1, markeredgewidth=1.0,
                mfc=ORACLE_COLOR, mec=ORACLE_COLOR, zorder=2)
        ax.step(weights, learned, where="post", color=color, ls="-", lw=1.1,
                marker=info["marker"], ms=2.8, markevery=1,
                mfc=color, mec=color, zorder=3)

        ax.set_title(f"$n={n}$", fontsize=13)
        ax.set_xlim(weights[0], weights[-1])
        ax.set_ylim(0.0, K * 1.03)
        ax.tick_params(labelsize=10.5, labelleft=True)
        ax.set_xlabel(r"Synthetic weight $\lambda$", fontsize=11)
        if pi == 0:
            ax.set_ylabel(r"Synthetic sample size $k$", fontsize=12)

    handles = [
        Line2D([0], [0], color=color, ls="-", lw=1.1,
               marker=info["marker"], ms=4.0, mfc=color, mec=color,
               label=r"Learned frontier $\hat{k}(\lambda)$"),
        Line2D([0], [0], color=ORACLE_COLOR, ls="-", lw=1.0,
               marker=ORACLE_MARKER, ms=4.5, markeredgewidth=1.0,
               mfc=ORACLE_COLOR, mec=ORACLE_COLOR,
               label=r"Oracle frontier $k^*(\lambda)$"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=11,
               frameon=False, bbox_to_anchor=(0.5, 0.02),
               handletextpad=0.6, columnspacing=1.8)

    fig.subplots_adjust(**SUBPLOTS_ADJUST)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"plot_frontier_{proxy}.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading checkpoints ...")
    data: dict = {}
    for proxy in ("full", "split"):
        for model in MODELS:
            for n in N_PANELS:
                data[(proxy, model, n)] = _load(proxy, model, n)
    n_splits = data[("full", MODELS[0], N_PANELS[0])]["k_hats"].shape[0]
    print(f"  loaded {len(data)} checkpoints ({n_splits} splits each)")

    for which in ("coverage", "width_reduction", "effective_size"):
        out = _make_metric_figure(which, data)
        print(f"  wrote {out.relative_to(HERE)}")

    for proxy in ("full", "split"):
        out = _make_frontier_figure(proxy, data)
        print(f"  wrote {out.relative_to(HERE)}")

    print("Done.")


if __name__ == "__main__":
    main()
