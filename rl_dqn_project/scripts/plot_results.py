"""Produce across-seed learning curves and a machine-readable summary."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_records(results_dir: Path, suite: str) -> dict[str, list[dict]]:
    """Load stored raw metrics; never synthesize missing training outcomes."""
    grouped = defaultdict(list)
    for file in sorted((results_dir / "metrics").glob("*_summary.json")):
        summary = json.loads(file.read_text(encoding="utf-8"))
        if suite != "all" and summary["suite"] != suite:
            continue
        run_id = summary["run_id"]
        eval_path = results_dir / "metrics" / f"{run_id}_evaluations.csv"
        with eval_path.open(newline="", encoding="utf-8") as handle:
            evaluation_rows = list(csv.DictReader(handle))
        grouped[summary["experiment"]].append({"summary": summary,
                                                 "evaluations": evaluation_rows})
    if not grouped:
        raise FileNotFoundError("No matching experiment results exist yet.")
    return dict(grouped)


def plot_evaluations(grouped: dict, figures_dir: Path, suffix: str) -> None:
    """Mean evaluation return and ±1 standard deviation across independent seeds."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for experiment, runs in sorted(grouped.items()):
        seed_series = []
        for run in runs:
            series = {int(row["env_steps"]): float(row["mean_return"])
                      for row in run["evaluations"] if row["kind"] == "periodic"}
            seed_series.append(series)
        common_steps = sorted(set.intersection(*(set(row) for row in seed_series)))
        if not common_steps:
            continue
        matrix = np.array([[series[step] for step in common_steps]
                           for series in seed_series], dtype=np.float64)
        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0, ddof=1) if len(runs) > 1 else np.zeros_like(mean)
        label = f"{experiment} (n={len(runs)})"
        line, = ax.plot(common_steps, mean, label=label)
        ax.fill_between(common_steps, mean - std, mean + std,
                        color=line.get_color(), alpha=0.15)
    ax.set(xlabel="Environment steps", ylabel="Mean greedy evaluation return",
           title=f"{suffix}: 10-episode periodic evaluations (±1 SD across seeds)")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / f"{suffix}_evaluation_curves.png", dpi=170)
    plt.close(fig)


def plot_training(grouped: dict, figures_dir: Path, suffix: str) -> None:
    """Visualize unaggregated training returns separately from evaluation."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for experiment, runs in sorted(grouped.items()):
        for number, run in enumerate(runs):
            run_id = run["summary"]["run_id"]
            file = figures_dir.parent / "metrics" / f"{run_id}_episodes.csv"
            with file.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            if not rows:
                continue
            steps = np.array([float(row["env_steps"]) for row in rows])
            rewards = np.array([float(row["train_return"]) for row in rows])
            # A rolling mean is purely for visualization, not evaluation.
            window = min(50, max(1, len(rewards) // 5))
            if window == 1:
                smooth = rewards
            else:
                smooth = np.convolve(rewards, np.ones(window) / window, mode="valid")
                steps = steps[window - 1:]
            ax.plot(steps, smooth, alpha=0.7, label=experiment if number == 0 else None)
    ax.set(xlabel="Environment steps", ylabel="Training episode return (rolling average)",
           title=f"{suffix}: training returns (not greedy evaluations)")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir / f"{suffix}_training_curves.png", dpi=170)
    plt.close(fig)


def write_summary(grouped: dict, output_dir: Path, suffix: str) -> None:
    """Compare final 100-episode results, per-seed variability and time."""
    rows = []
    for experiment, runs in sorted(grouped.items()):
        returns = np.array([r["summary"]["final_mean_return"] for r in runs])
        times = np.array([r["summary"]["wall_clock_seconds_total"] for r in runs])
        first_steps = [r["summary"]["first_provisional_10episode_threshold_step"]
                       for r in runs]
        rows.append({
            "experiment": experiment, "seed_count": len(runs),
            "final_100ep_return_mean_across_seeds": float(returns.mean()),
            "final_100ep_return_sd_across_seeds": (float(returns.std(ddof=1))
                                                   if len(runs) > 1 else 0.0),
            "wall_clock_seconds_mean": float(times.mean()),
            "first_provisional_threshold_steps_per_seed": json.dumps(first_steps),
            "final_100ep_threshold_pass_count": sum(
                r["summary"]["reached_threshold_final_100episodes"] is True for r in runs),
        })
    with (output_dir / f"{suffix}_aggregate.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot measured RL data across seeds.")
    parser.add_argument("--results-dir", default="results/full")
    parser.add_argument("--suite", choices=("tier1", "tier2", "tier3", "all"), default="all")
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    grouped = load_records(results_dir, args.suite)
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    # Separate environments because CartPole and Acrobot returns are not
    # directly comparable. In "all" mode this also plots vanilla and Double
    # DQN on the SAME Acrobot axes for the Tier 3 comparison.
    by_environment = defaultdict(dict)
    for name, runs in grouped.items():
        env_id = runs[0]["summary"]["env_id"]
        by_environment[env_id][name] = runs
    for env_id, env_group in by_environment.items():
        suffix = f"{args.suite}_{env_id.lower().replace('-', '_')}"
        plot_evaluations(env_group, figures_dir, suffix)
        plot_training(env_group, figures_dir, suffix)
        write_summary(env_group, figures_dir, suffix)
    print(f"Figures and aggregate CSV written to {figures_dir}")


if __name__ == "__main__":
    main()
