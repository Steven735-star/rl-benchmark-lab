# RL Benchmark Lab

**A reproducible benchmark of tabular and deep reinforcement learning agents across Gymnasium environments.**

RL Benchmark Lab is a Python and PyTorch project for implementing, training, evaluating, and comparing value-based reinforcement learning algorithms. It provides a common experimental workflow for **tabular Q-learning**, **Deep Q-Networks (DQN)**, and **Double DQN**, with an emphasis on experimental controls, reproducibility, and transparent reporting of both successful and unsuccessful training runs.

The current benchmark covers **CartPole-v1** and **Acrobot-v1** from Gymnasium. It investigates how state discretization, exploration schedules, learning rates, and target-network design affect sample efficiency, training stability, and final policy performance.

> **Experimental status:** The accompanying scientific report is a working draft. The results below were extracted from an audited archive of actual experiments; they should not be interpreted as evidence of consistent convergence. No tested configuration reached its target on all three random seeds.

## Contents

- [Research objectives](#research-objectives)
- [Algorithms and environments](#algorithms-and-environments)
- [Experimental design](#experimental-design)
- [Results](#results)
- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Running experiments](#running-experiments)
- [Reproducibility and outputs](#reproducibility-and-outputs)
- [Limitations and future work](#limitations-and-future-work)
- [Technical references](#technical-references)

## Research objectives

The benchmark addresses three questions:

1. **Tabular vs. deep RL:** How do alternative Q-learning discretizations compare with DQN under an equal environment-step budget on CartPole?
2. **DQN sensitivity:** How do learning rate, epsilon-decay duration, and the presence of a target network influence DQN performance on Acrobot?
3. **Value-estimation variants:** Does Double DQN improve final returns and consistency relative to a vanilla DQN baseline under matched experimental conditions?

The project prioritizes comparable measurements over presenting a single best training run. It reports variation across random seeds, distinguishes training returns from greedy evaluation returns, and treats intermediate threshold crossings as provisional rather than final success.

## Algorithms and environments

| Method | Environment | Core idea | Experimental role |
| --- | --- | --- | --- |
| Tabular Q-learning | `CartPole-v1` | Discretized observations and an explicit Q-table | Compare 6, 10, and 20 bins per observation dimension |
| DQN | `CartPole-v1` | Neural Q-function with replay memory and a target network | Compare against tabular agents using the same training-step budget |
| DQN | `Acrobot-v1` | Replay-based neural value learning | Study learning rate and exploration-decay sensitivity |
| DQN without target network | `Acrobot-v1` | Online-network bootstrap target | Isolate the contribution of the target network |
| Double DQN | `Acrobot-v1` | Separate next-action selection and target evaluation | Compare against vanilla DQN on matched seeds |

**Environment properties**

| Environment | Observation space | Action space | Project success criterion |
| --- | --- | --- | --- |
| `CartPole-v1` | 4 continuous values | 2 discrete actions | Mean greedy return **>= 475** over 100 final episodes |
| `Acrobot-v1` | 6 continuous values | 3 discrete actions | Mean greedy return **>= -100** over 100 final episodes |

These thresholds are from the project specification and are not necessarily the default Gymnasium reward thresholds.

### Implementation details

**Tabular Q-learning**

- Uniform discretization with 6, 10, or 20 bins per observation dimension.
- Explicit clipping ranges for CartPole position, velocity, pole angle, and angular velocity.
- Learning rate `0.1`, discount factor `0.99`, zero-initialized Q-values.
- Epsilon-greedy exploration, decaying from `1.0` to `0.05` over 80% of training steps.

**DQN**

- Fully connected Q-network with two hidden ReLU layers of width 128.
- Experience replay with capacity `50,000` and minibatches of `64`.
- Separate target network, normally synchronized every `1,000` steps.
- Adam optimizer, default learning rate `5e-4`, discount factor `0.99`.
- Learning warmup of `1,000` transitions, optimization every 4 environment steps, and gradient-norm clipping at 10.
- Epsilon-greedy exploration, normally decaying from `1.0` to `0.05` over 50% of training steps.

**Double DQN** uses the online Q-network to select the next action and the target Q-network to evaluate it. The goal is to reduce maximization-related value overestimation; the current experiments did **not** log value-estimation error directly, so the underlying mechanism was not empirically verified.

The implementation differentiates between **environment termination** and **time-limit truncation** when calculating bootstrap targets.

## Experimental design

The evaluated benchmark consists of **11 configurations x 3 independent seeds (`0`, `1`, `2`) = 33 training runs**.

| Suite | Environment | Configurations | Runs | Steps per run | Total training steps |
| --- | --- | ---: | ---: | ---: | ---: |
| Tier 1 | CartPole-v1 | 4 | 12 | 200,000 | 2,400,000 |
| Tier 2 | Acrobot-v1 | 6 | 18 | 100,000 | 1,800,000 |
| Tier 3 | Acrobot-v1 | 1 | 3 | 100,000 | 300,000 |
| **Total** | | **11** | **33** | | **4,500,000** |

All runs follow the same evaluation protocol:

- **Independent seeds:** 0, 1, and 2 for every configuration.
- **Periodic evaluation:** 10 greedy episodes (`epsilon = 0`) every 10,000 environment steps.
- **Final evaluation:** 100 greedy episodes per run.
- **Aggregated statistics:** mean and *sample* standard deviation of the three per-seed final evaluation means.
- **Learning curves:** environment steps on the x-axis, cross-seed mean and +/- 1 sample standard deviation where available.
- **Success definition:** only the 100-episode final evaluation determines whether a run meets the specified threshold.

For Tier 2, two hyperparameter families are varied **one factor at a time** around the same baseline:

- Learning rate: `1e-4`, `5e-4` (baseline), `1e-3`.
- Epsilon decay: 20%, 50% (baseline), 80% of training steps.

The target-network ablation and Double DQN comparison retain the Acrobot environment and the matched step budget and seeds.

## Results

All numbers in this section come from `Informe_y_analisis_RL/consolidado_33_ejecuciones.csv` and the associated scientific review. They summarize **final 100-episode greedy evaluations**, not the best interim checkpoint or the average training episode.

### Tier 1: CartPole — tabular Q-learning vs. DQN

| Configuration | Mean return | Sample SD | Successful seeds |
| --- | ---: | ---: | ---: |
| Q-learning, 6 bins | 161.80 | 37.78 | 0 / 3 |
| Q-learning, 10 bins | 275.68 | 193.52 | 1 / 3 |
| **Q-learning, 20 bins** | **280.87** | 127.96 | 0 / 3 |
| DQN | 117.46 | 34.45 | 0 / 3 |

Tabular Q-learning with 20 bins achieved the highest average final return, but its advantage over 10 bins is small compared with seed-to-seed variability. The 10-bin variant reached 497.22 on one seed but did not reproduce that outcome on the other two. DQN showed promising intermediate performance, followed by weaker final evaluation results.

**Figures:** [CartPole learning curves](Informe_y_analisis_RL/paper_figures/cartpole_evaluations.pdf) · [Final returns](Informe_y_analisis_RL/paper_figures/cartpole_final.pdf)

### Tier 2: Acrobot — DQN sensitivity and ablation

| Configuration | Mean return | Sample SD | Successful seeds |
| --- | ---: | ---: | ---: |
| DQN baseline (`lr=5e-4`) | -371.05 | 223.35 | 0 / 3 |
| **Learning rate `1e-4`** | **-254.25** | 217.55 | 1 / 3 |
| Learning rate `1e-3` | -373.04 | 219.90 | 0 / 3 |
| Epsilon decay over 20% | -500.00 | 0.00 | 0 / 3 |
| Epsilon decay over 80% | -361.72 | 239.51 | 1 / 3 |
| Without target network | -500.00 | 0.00 | 0 / 3 |

**Higher returns are better**, including on Acrobot where scores are negative. The `1e-4` learning-rate configuration obtained the best Tier 2 mean, although it remained inconsistent across seeds. Removing the target network produced a final return of -500 on all three seeds in this experiment; that observation supports a stabilizing role *in this setup*, not a universal causal conclusion.

**Figures:** [Acrobot learning curves](Informe_y_analisis_RL/paper_figures/acrobot_evaluations.pdf) · [Final returns](Informe_y_analisis_RL/paper_figures/acrobot_final.pdf)

### Tier 3: Double DQN vs. vanilla DQN

| Seed | Vanilla DQN | Double DQN | Difference |
| --- | ---: | ---: | ---: |
| 0 | -113.15 | -213.49 | -100.34 |
| 1 | -500.00 | -94.64 | +405.36 |
| 2 | -500.00 | -500.00 | 0.00 |
| **Mean** | **-371.05** | **-269.38** | **+101.67** |

Double DQN improved the cross-seed mean by 101.67 points, but the paired results were highly variable: one seed worsened, one improved substantially, and one remained unchanged. Three seeds are insufficient to establish a statistically robust general improvement.

**Figure:** [Paired Double DQN comparison](Informe_y_analisis_RL/paper_figures/paired_double_dqn.pdf)

### Overall observations

- **4 of 33 individual runs** passed their final project-specific threshold.
- **No configuration** passed on all three random seeds.
- The run logs record **1,229.97 seconds (approximately 20.5 minutes)** of summed per-run time, including evaluations but excluding auxiliary tasks such as plotting and GIF generation. This is **not** an end-to-end benchmark runtime.
- Failed or unstable runs are retained as experimental evidence rather than omitted from the analysis.

For additional context, see the [scientific review](Informe_y_analisis_RL/REVISION_CIENTIFICA_ES.md), [detailed evaluations](Informe_y_analisis_RL/detalle_33_ejecuciones.csv), [aggregated results](Informe_y_analisis_RL/consolidado_33_ejecuciones.csv), and [audit metadata](Informe_y_analisis_RL/auditoria.json).

## Repository structure

```text
rl-benchmark-lab/
├── README.md                          # Main repository documentation
├── .gitignore
├── rl_dqn_project/                   # Training and evaluation application
│   ├── README.md
│   ├── GUIA_ES.md
│   ├── requirements.txt
│   ├── requirements-lock.txt
│   ├── agents/
│   │   ├── tabular.py                # Tabular Q-learning agent
│   │   └── dqn.py                    # DQN and Double DQN implementation
│   ├── envs/
│   │   └── discretizer.py            # State discretization
│   ├── experiments/
│   │   ├── configs.py                # Experiment configurations
│   │   └── trainer.py                # Training and evaluation workflow
│   ├── scripts/
│   │   ├── run_experiments.py         # Execute benchmark suites
│   │   ├── plot_results.py           # Generate performance figures
│   │   └── record_agent.py           # Record trained agents
│   ├── utils/
│   │   ├── evaluation.py
│   │   └── seeding.py
│   ├── tests/                         # Automated tests
│   ├── report/                        # Supplementary report material
│   └── results/
│       ├── pilot_tier1/              # Initial CartPole runs
│       ├── pilot_tier2/              # Initial Acrobot runs
│       └── full/                     # Full results when extracted/generated
└── Informe_y_analisis_RL/           # Scientific analysis and manuscript
    ├── REVISION_CIENTIFICA_ES.md
    ├── articulo_RL_DQN_borrador.tex
    ├── articulo_RL_DQN_borrador.pdf
    ├── auditoria.json
    ├── consolidado_33_ejecuciones.csv
    ├── detalle_33_ejecuciones.csv
    └── paper_figures/
```

The `results/full/` folder represents the expected location for the complete benchmark results, **not a guarantee that those files are committed to Git**. The local project was reported to occupy approximately 3 GB, including large archives and experiment artifacts, which may need separate distribution. The paper's summary tables alone do not replace full per-run logs or checkpoints.

## Installation

### Requirements

- Python and `pip` (use the versions supported by `rl_dqn_project/requirements.txt` and the lockfile).
- NumPy, PyTorch, Gymnasium, Matplotlib, and other packages specified by the project.
- A CPU-capable computer; the reported experiments were executed on CPU.
- Git for source control.

The archived experiment metadata records **Python 3.14.4**, **Gymnasium 1.4.0**, **PyTorch 2.15.0+cu132**, and `device=cpu`. These are recorded provenance values, not a claim that every platform requires the identical build.

```bash
# Clone the repository
git clone git@github.com:Steven735-star/rl-benchmark-lab.git
cd rl-benchmark-lab/rl_dqn_project

# Create and activate an isolated Python environment
python3 -m venv .venv
source .venv/bin/activate

# Install project dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, use `.venv\Scripts\activate` instead of the Unix activation command. If platform-specific dependency versions differ, consult `requirements-lock.txt` and the Python/PyTorch compatibility requirements before attempting exact reproduction.

### Run tests

From the `rl_dqn_project/` directory:

```bash
python -m pytest tests/
```

If `pytest` is not included in the main requirements file, install it into the active environment before running the tests.

## Running experiments

Run commands **from the `rl_dqn_project/` directory**, where the `scripts`, `agents`, and `experiments` packages reside.

```bash
# Tier 1: CartPole tabular Q-learning vs. DQN
python -m scripts.run_experiments --suite tier1

# Tier 2: Acrobot DQN, parameter sweeps, and target-network ablation
python -m scripts.run_experiments --suite tier2

# Tier 3: Double DQN comparison
python -m scripts.run_experiments --suite tier3

# Generate benchmark plots after the experiment suites finish
python -m scripts.plot_results --suite all
```

These suite commands follow the accompanying manuscript. Check the available arguments against your checked-out version of `scripts/run_experiments.py` and `scripts/plot_results.py`. To inspect additional options, try:

```bash
python -m scripts.run_experiments --help
python -m scripts.plot_results --help
python -m scripts.record_agent --help
```

The manuscript specifies seeds `0`, `1`, and `2` by default and records full results under `results/full/`. Rerunning a benchmark can take a different amount of time and produce slightly different outcomes depending on software versions, hardware, and nondeterministic operations.

### Scientific article

The manuscript source is located at `Informe_y_analisis_RL/articulo_RL_DQN_borrador.tex`. Compile it from that folder so its relative figure paths resolve:

```bash
cd ../Informe_y_analisis_RL
pdflatex articulo_RL_DQN_borrador.tex
pdflatex articulo_RL_DQN_borrador.tex
```

A working LaTeX installation with the packages referenced by the manuscript is required. The PDF is a draft and still requires final author and reference review.

## Reproducibility and outputs

The benchmark records or expects the following categories of artifacts:

| Artifact | Typical format | Purpose |
| --- | --- | --- |
| Training episodes | `.csv` | Training return, steps, and episode history |
| Periodic evaluations | `.csv` | Greedy returns at comparable environment-step checkpoints |
| Run summaries/configurations | `.json` | Final metrics, seed, environment, hyperparameters, software and timings |
| Trained policy checkpoints | `.pt` / `.npz` | Reload trained DQN and tabular policies |
| Figures | `.pdf` / `.png` | Learning curves and final-return comparisons |
| Recorded agent trajectories | `.gif` / video | Qualitative illustrations of learned behavior |
| Scientific report | `.tex` / `.pdf` | Methods, results, interpretation, and analysis-only discussion |

**Important:** The reported 33-run audit was performed using a complete results archive that included per-run CSVs, JSON summaries, 33 model checkpoints, eight PNG figures, and two agent GIFs. If large compressed archives or model checkpoints are excluded via `.gitignore`, those items **will not automatically be downloadable from GitHub**. For complete archival reproducibility, publish the necessary uncompressed logs/checkpoints using Git LFS, a release asset, or another clearly linked storage location; alternatively, regenerate them using the experiment scripts.

This repository should not commit virtual environments, Python caches, temporary LaTeX compilation files, or duplicate ZIP archives. It should retain source code, environment specifications, experiment configurations, relevant tables, and the figures referenced by the article.

## Limitations and future work

The current results have important limitations:

- Only **three independent training seeds** were used, limiting statistical certainty.
- Periodic evaluations use **10 episodes**, while final evaluations use **100 episodes**; the two must not be treated as equivalent.
- Only the **last training checkpoint** was retained, meaning a promising intermediate policy could later degrade.
- Loss curves, replay-buffer diagnostics, and Q-value estimation errors were **not recorded**, so explanations for instability remain hypotheses.
- The implemented environments have **discrete actions**. Continuous-action locomotion tasks such as BipedalWalker and MuJoCo benchmarks are considered in written analysis rather than implemented using ordinary DQN.

Potential extensions include best-checkpoint selection with a separate held-out evaluation set, additional random seeds, richer training diagnostics, target-update studies, prioritized replay, and other value-based variants. These are **future directions**, not completed benchmark features.

## Implementation integrity and AI disclosure

The reinforcement learning agents are implemented using Python and PyTorch rather than imported as ready-made agents from a third-party RL framework. Generative AI assistance was used in planning, code development, debugging, experimental interpretation, visualization, and report drafting. The corresponding statement is documented in `rl_dqn_project/report/ai_use_statement.md` and the scientific manuscript.

All reported values are presented as measurements from the audited experiment archive; the current documentation does not claim that every run was successful or that Double DQN produced a statistically significant benefit. Further changes to the repository should preserve the distinction between **measured results**, **interpretations**, and **proposed future experiments**.

## Technical references

- [Gymnasium documentation](https://gymnasium.farama.org/)
- [PyTorch documentation](https://pytorch.org/docs/stable/index.html)
- Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed., 2018).
- Mnih et al., *Human-level control through deep reinforcement learning*, *Nature* (2015).
- van Hasselt, Guez & Silver, *Deep Reinforcement Learning with Double Q-learning*, AAAI (2016).

---

*RL Benchmark Lab — a transparent, reproducible comparison of reinforcement learning methods and their observed failure modes.*
