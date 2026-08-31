# Learning a Size–Weight Frontier for Synthetic-Augmented Inference

This repository contains the code for implementing the algorithms and reproducing the experiments in the following paper:

Huang, Chengpiao and Wang, Kaizheng. (2026). Learning a Size–Weight Frontier for Synthetic-Augmented Inference. Available at arXiv: https://arxiv.org/abs/2608.28576 and SSRN: https://ssrn.com/abstract=7365498.



## Method overview

Synthetic data can help statistical inference when real data are scarce, but naively treating imperfect simulations as real samples introduces bias and can invalidate conclusions. We develop a general framework that augments real data with synthetic data to form statistically valid confidence interval. Our framework is characterized by the number of synthetic samples and the weight assigned to each. We propose algorithms to learn a **size–weight frontier** $\hat{k}(\lambda)$: for each weight $\lambda \in [0, 1]$ assigned to synthetic observations, it gives the largest number of synthetic samples $k$ that can be combined with $n$ real samples while still maintaining valid $1 - \alpha$ confidence-interval (CI) coverage. The experiments use large-language-model responses to augment **WorldValuesBench** opinion-survey data.

For every configuration of real sample size $n$, level $\alpha$, CI type, and model, we repeatedly split the questions into a calibration set and a test set. On the **calibration** set we form a combined confidence interval for each question's mean from $n$ real samples plus $k$ synthetic samples weighted by $\lambda$, and estimate the empirical coverage grid $\hat{p}_n(\lambda, k)$ by checking how often a proxy CI for the unknown true mean is contained in it; a non-increasing envelope of this grid yields the learned frontier $\hat{k}(\lambda)$. On the **test** set we compare against the oracle frontier (built from the full-population means) and measure coverage, CI-width reduction, and effective synthetic sample size $\lambda \cdot \hat{k}(\lambda)$. Results are averaged over `NUM_SPLITS` splits.

## Repository layout

```
.
├── core.py                   # real-synthetic combined CI, proxy CI, coverage grid, frontier
├── data_loaders.py           # WorldValuesBench loader + DATASET_CONFIGS
├── experiments.py            # calibration/test data split + checkpointing
├── plots.py                  # builds the paper figures from the checkpoints
├── run_worldvalue.py         # entry point: computes checkpoints (one proxy per run)
├── data/
│   └── worldvalue/           # input data (see Data setup)
└── results/                  # created / updated at run time
    ├── worldvalue_full50/    # full-proxy run  (gamma = 0.5)
    ├── worldvalue_split50/   # split-proxy run (gamma = 0.5)
    └── figures/              # paper figures (built by plots.py)
```


## Installation

Requires Python 3.10+.

```bash
pip install numpy scipy pandas matplotlib tqdm
```


## Data setup

**Data source.** The underlying survey data is the **WorldValuesBench** dataset (Zhao et al., *WorldValuesBench: A Large-Scale Benchmark Dataset for Multi-Cultural Value Awareness of Language Models*, LREC-COLING 2024), which is curated from the World Values Survey (Haerpfer et al., *World Values Survey: Round Seven — Country-Pooled Datafile (2017–2020)*, 2020). The preprocessed real responses and the synthetic LLM responses used in this repository are taken from Iyengar et al. (*Model-Free Assessment of Simulator Fidelity via Quantile Curves*, arXiv:2512.05024, 2025), and are available from their GitHub repository: https://github.com/yushiouwillylin/simulator_fidelity.

Place the WorldValuesBench data files in `data/worldvalue/`:

```
data/worldvalue/
├── retained_questions_235.json
├── full_population_response_clean.pkl
└── synthetic answers/
    └── clean/
        ├── synthetic_answers_clean_gpt4o.pkl
        ├── synthetic_answers_clean_gpt-5-mini.pkl
        └── ...     # one pickle per LLM
```

Real responses are drawn from the full-population pool (~96K per question), which also serves as the ground truth for evaluating test coverage. Each synthetic file holds 500 LLM responses per question; responses are already mapped to $[-1, 1]$.


## Usage

Run from this directory:

```bash
python run_worldvalue.py
```

Pick the proxy scheme at the top of `run_worldvalue.py` via `PROXY_METHOD` (`"full"` or `"split"`) and `GAMMA`. Each run computes checkpoints for one proxy and writes to its own output directory. After running both proxies, build all paper figures:

```bash
python plots.py
```


## Configuration

The configuration and hyperparameters are set at the top of `run_worldvalue.py`:

| Variable        | Meaning                                              |
|-----------------|------------------------------------------------------|
| `PROXY_METHOD`  | `"full"` or `"split"` proxy                          |
| `GAMMA`         | Unified proxy constant $\gamma \in (0, 1)$           |
| `MODELS`        | Which LLM simulators to evaluate                     |
| `N_VALUES`      | Real sample sizes to sweep over                      |
| `ALPHA_VALUES`  | Target miscoverage levels (e.g. `0.10`, `0.15`)      |
| `NUM_SPLITS`    | Number of random train/test splits to average over   |
| `CI_TYPES`      | `"bernstein"`, `"hoeffding"`, or `"clt"`             |
| `WEIGHTS_GRID`  | The $\lambda$ grid                                   |
| `K_GRID`        | The $k$ grid                                         |

Dataset-specific constants (`response_range`, `param_range`, `n_max`) are set in `data_loaders.DATASET_CONFIGS`.

Checkpoints are named `{model}_n{n}_alpha{aa}_{ci}_gamma{gg}_nmax{nmax}[_split].pkl`. The runner detects incompatible grids automatically and recomputes; to force a full recomputation, delete the matching pickles in `results/worldvalue_<proxy>50/checkpoints/`.


## Outputs

`run_worldvalue.py` writes, for each proxy, to `results/worldvalue_<proxy>50/`:

- `checkpoints/` — one `.pkl` per configuration
- `summary_<ci>.csv` — summary statistics used to plot the figures

`plots.py` then reads those checkpoints and writes the figures to `results/figures/`. All are three-panel ($n = 10, 30, 50$), with $\alpha = 0.1$ and the empirical Bernstein CI:

| Figure | Content |
|--------|---------|
| `plot_coverage.pdf`        | Test coverage at the learned frontier; full vs split proxy, one curve per model |
| `plot_width_reduction.pdf` | Relative CI-width reduction $R_n(\lambda)$; full vs split proxy                  |
| `plot_effective_size.pdf`  | Effective synthetic sample size $\lambda \cdot \hat{k}(\lambda)$; full vs split proxy |
| `plot_frontier_full.pdf`   | GPT-4o learned frontier $\hat{k}(\lambda)$ vs oracle $\hat{k}^*(\lambda)$ (one seed), full proxy |
| `plot_frontier_split.pdf`  | GPT-4o learned frontier $\hat{k}(\lambda)$ vs oracle $\hat{k}^*(\lambda)$ (one seed), split proxy    |


## Citation

```bibtex
@article{HWa26,
  title  = {Learning a Size--Weight Frontier for Synthetic-Augmented Inference},
  author = {Huang, Chengpiao and Wang, Kaizheng},
  journal = {arXiv preprint arXiv:2608.28576},
  year   = {2026},
}
```
