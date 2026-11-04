# Project 01 — Tabular RL, DQN, and Double DQN

**Project scope:** CartPole-v1 for tabular Q-learning versus DQN (Tier 1), Acrobot-v1 for DQN hyperparameter sweeps and a target-network ablation (Tier 2), and Double DQN on Acrobot-v1 (Tier 3, required for a two-person team).

This repository implements the agents directly in Python/NumPy/PyTorch. Gymnasium supplies only the environments. It does **not** import a ready-made DQN from Stable-Baselines3 or another RL framework.

## Installation

Use **Python 3.10+** and a laptop with a CPU. From the repository root:

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest -q
```

For a fast **end-to-end smoke run** against real Gymnasium environments:

```bash
python -m scripts.run_experiments --suite smoke
python -m scripts.plot_results --results-dir results/smoke --suite all
```

Smoke uses one seed, 200 environment steps per configuration, and very small evaluation batches. **Never report smoke outcomes as project findings.**

## Full experimental protocol

Three independent seeds (0, 1, 2) are used by default. The independent evaluation environment uses epsilon=0, 10 greedy episodes at each checkpoint, and 100 greedy episodes at final evaluation. Evaluation states do not affect training exploration randomness.

**Tier 1 (four configurations at 200,000 environment steps per seed):**

```bash
python -m scripts.run_experiments --suite tier1 --results-dir results/full
python -m scripts.plot_results --results-dir results/full --suite tier1
```

**Tier 2 (six configurations at 100,000 environment steps per seed):**

```bash
python -m scripts.run_experiments --suite tier2 --results-dir results/full
python -m scripts.plot_results --results-dir results/full --suite tier2
```

**Tier 3 (Double DQN, three seeds, same Acrobot step budget as Tier 2):**

```bash
python -m scripts.run_experiments --suite tier3 --results-dir results/full
python -m scripts.plot_results --results-dir results/full --suite all
```

**Everything in one command** (21 runs on Acrobot + 12 on CartPole = 33 runs):

```bash
python -m scripts.run_experiments --suite all --results-dir results/full
python -m scripts.plot_results --results-dir results/full --suite all
```

**Warning on computing budget:** Running all 33 training jobs, plus the associated evaluations, may exceed the instructor's ~4–6 CPU-hour target on a slower laptop. Time a shorter pilot before selecting full training/evaluation budgets. For pilot runs only, use:

```bash
python -m scripts.run_experiments --suite tier1 --steps-tier1 10000 --eval-every 2000 --final-eval-episodes 10 --results-dir results/pilot_tier1
```

Do not mix pilot and final results or present different step budgets as a fair comparison. **No agent success is guaranteed.** Failure can be analyzed and reported in the paper.

## Make report figures

Each command above produces the following artifacts for the corresponding suite:

- `results/full/metrics/<experiment>_seed<seed>_episodes.csv`: raw training returns.
- `results/full/metrics/<experiment>_seed<seed>_evaluations.csv`: checkpoint and final evaluations.
- `results/full/metrics/<experiment>_seed<seed>_summary.json`: final 100-episode return, hyperparameters, hardware and duration.
- `results/full/models/<experiment>_seed<seed>.npz` or `.pt`: saved policy weights.
- `results/full/figures/<suite>_<environment>_evaluation_curves.png`: evaluation learning curves, mean ± SD across independent seeds.
- `results/full/figures/<suite>_<environment>_training_curves.png`: **separate** training curves.
- `results/full/figures/<suite>_<environment>_aggregate.csv`: cross-seed outcome/variance/time summary.

The `--suite all` plot command separates environments but includes the Tier 2 vanilla baseline **and** Tier 3 Double DQN together on Acrobot axes.

The periodic 10-episode threshold crossing is **provisional**. True project success is assessed from the final 100-episode evaluation per seed; the final summary contains `reached_threshold_final_100episodes`.

To create a video/GIF from **actual trained models**:

```bash
python -m scripts.record_agent \
  --checkpoint results/full/models/cartpole_dqn_seed0.pt \
  --env-id CartPole-v1 \
  --output results/full/videos/cartpole_dqn_seed0.gif

python -m scripts.record_agent \
  --checkpoint results/full/models/acrobot_double_dqn_seed0.pt \
  --env-id Acrobot-v1 \
  --output results/full/videos/acrobot_double_dqn_seed0.gif
```

## Experimental controls and design choices

- Tabular discretization uses 6, 10, or 20 bins per dimension. The finite CartPole cutoffs are ±2.4, ±3.0, ±0.2095, and ±3.5; values outside them are clipped.
- Default DQN uses a two-hidden-layer (128, 128) ReLU network, Adam (learning rate `5e-4`), replay capacity 50,000, batch size 64, gamma `0.99`, learning starts after 1,000 environment steps, one gradient update every 4 steps, and hard target synchronization every 1,000 steps.
- Tier 2 varies learning rate (`1e-4`, `5e-4`, `1e-3`) and epsilon decay fraction (`0.2`, `0.5`, `0.8`) one factor at a time around the shared baseline.
- Target-network ablation replaces the target-network estimate with the current online network for bootstrap targets; replay is unchanged.
- Tier 3 uses Double DQN's online-argmax, target-network evaluation; compare it with the vanilla DQN baseline from Tier 2 under identical seeds and step budget.
- Gymnasium `terminated` disables Bellman bootstrapping. `truncated` resets the rollout but **does not** disable Bellman bootstrapping. This distinction avoids an important common RL implementation error.
- Final evaluation mean uses 100 episodes; all raw data are retained and separately labeled.

## Reproducibility and caveats

The program stores seeds, experiment names, hyperparameters and recorded wall-clock times. PyTorch's deterministic algorithm mode is requested. Exact floating-point reproducibility across CPU architectures, PyTorch versions and operating systems is not guaranteed; use one pinned environment for comparisons, and record versions with `pip freeze > requirements-lock.txt`.

This repository has **no fabricated training curves, benchmarks, or learned weights**. Report statistics only after running the scripts on the actual machine. See `GUIA_ES.md`, `report/paper_outline.md`, and `report/ai_use_statement.md`.

## Analysis-only component

A separately drafted, citation-oriented theoretical analysis is available in `report/analysis_only_draft.md`. It contains a quantified comparison of action spaces after discretization, an explanation of PPO/SAC/TD3, and an argument against tabular Atari-pixel methods. Review and cite the original papers when preparing the final double-column report.
