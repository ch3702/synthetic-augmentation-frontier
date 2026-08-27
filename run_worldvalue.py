"""
run_worldvalue.py
=================
Entry point for running the synthetic augmentation size-weight frontier on the
WorldValuesBench dataset (World Values Survey), using several LLMs as
synthetic simulators.

Usage (run from this directory):

    python run_worldvalue.py

Outputs are written under results/worldvalue_<proxy><gamma>/.

The WorldValuesBench data is read from data/worldvalue/.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mplconfig"))

from data_loaders import DATASET_CONFIGS, load_worldvalue
from experiments import run_all_experiments


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Proxy scheme for learning the frontier, both governed by the unified constant gamma in (0, 1)
# "full" uses real[:n_max] at coverage 1 - gamma*alpha
# "split" uses the disjoint tail real[n:n_max] at coverage 1 - gamma
PROXY_METHOD = "split"      # "full" | "split"
GAMMA        = 0.5          # unified proxy constant

OUTPUT_DIR = (HERE / "results" / f"worldvalue_{PROXY_METHOD}{int(GAMMA*100):02d}")

# LLM simulators to evaluate
# Must be keys of data_loaders.WORLDVALUE_FILE_BY_MODEL
MODELS = [
    "gpt4o",
    "gpt-5-mini",
    # "llama-3.3",
    # "Qwen3-235B",
    # "uniform",
]

# Sweep over real sample sizes and miscoverage levels.
N_VALUES     = [10, 30, 50]
ALPHA_VALUES = [0.10]

# Experimental knobs
NUM_SPLITS   = 50
CI_TYPES     = ["bernstein"]
WEIGHTS_GRID = np.round(np.arange(0.0, 1.05, 0.05), 2)
K_GRID       = np.arange(1, 201, dtype=int)
MASTER_SEED  = 42


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = DATASET_CONFIGS["worldvalue"]
    n_max          = cfg["n_max"]
    response_range = cfg["response_range"]
    param_range    = cfg["param_range"]

    print("=" * 60)
    print("Synthetic Augmentation Frontier  —  WorldValuesBench")
    print("=" * 60)
    print(f"Output dir  : {OUTPUT_DIR}")
    print(f"models      : {MODELS}")
    print(f"n values    : {N_VALUES}")
    print(f"alpha       : {ALPHA_VALUES}")
    print(f"ci types    : {CI_TYPES}")
    print(f"proxy       : {PROXY_METHOD}" + (f" (gamma={GAMMA})" if PROXY_METHOD == "split" else ""))
    print(f"splits      : {NUM_SPLITS}")
    print()

    print("Loading data ...")
    real_answers, all_syn_answers = load_worldvalue(MODELS)
    print(f"  {len(real_answers)} questions loaded")

    run_all_experiments(
        real_answers   = real_answers,
        all_syn_answers= all_syn_answers,
        output_dir     = OUTPUT_DIR,
        models         = MODELS,
        n_values       = N_VALUES,
        alpha_values   = ALPHA_VALUES,
        ci_types       = CI_TYPES,
        response_range = response_range,
        param_range    = param_range,
        num_splits     = NUM_SPLITS,
        weights_grid   = WEIGHTS_GRID,
        k_grid         = K_GRID,
        n_max          = n_max,
        master_seed    = MASTER_SEED,
        proxy_method   = PROXY_METHOD,
        gamma          = GAMMA,
    )

    # Figures are built separately (after running both proxies) by plots.py.
    print("\nDone.  Run `python plots.py` to build the paper figures.")
