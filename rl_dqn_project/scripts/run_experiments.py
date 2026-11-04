"""Command-line entry point for all experiments and short integration tests."""
import argparse
import json
from pathlib import Path

import torch

from agents.dqn import DQNConfig
from experiments.configs import get_experiments
from experiments.trainer import train_one


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reproduce Tier 1/2/3 RL experiments.")
    parser.add_argument("--suite", choices=("tier1", "tier2", "tier3", "all", "smoke"),
                        default="all")
    parser.add_argument("--results-dir", default="results/full")
    parser.add_argument("--steps-tier1", type=int, default=200_000)
    parser.add_argument("--steps-tier2", type=int, default=100_000)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--eval-every", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--final-eval-episodes", type=int, default=100)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--learning-starts", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-width", type=int, default=128)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if len(args.seeds) != len(set(args.seeds)):
        raise ValueError("Seed values must be unique.")
    torch.set_num_threads(args.torch_threads)
    try:
        import gymnasium
    except ImportError as error:
        raise SystemExit("Install the declared requirements first: pip install -r requirements.txt") from error

    # Smoke mode intentionally differs from graded runs: four tiny test jobs,
    # one seed, very short training, and tiny evaluation batches.
    if args.suite == "smoke":
        specs = [spec for spec in get_experiments("all") if spec.name in {
            "cartpole_tabular_6bins", "cartpole_dqn", "acrobot_dqn_default", "acrobot_double_dqn"}]
        seeds = [0]
        eval_every, eval_episodes, final_episodes = 100, 2, 3
        steps_tier1 = steps_tier2 = 200
        dqn_config = DQNConfig(learning_starts=16, batch_size=8, replay_capacity=500,
                               hidden_width=32, target_update_frequency=50)
        output_dir = Path("results/smoke") if args.results_dir == "results/full" else Path(args.results_dir)
    else:
        specs = get_experiments(args.suite)
        seeds = args.seeds
        eval_every, eval_episodes, final_episodes = (args.eval_every,
                                                     args.eval_episodes, args.final_eval_episodes)
        steps_tier1, steps_tier2 = args.steps_tier1, args.steps_tier2
        dqn_config = DQNConfig(learning_starts=args.learning_starts,
                               batch_size=args.batch_size, hidden_width=args.hidden_width)
        output_dir = Path(args.results_dir)

    summaries = []
    for spec in specs:
        steps = steps_tier1 if spec.suite == "tier1" else steps_tier2
        for seed in seeds:
            summaries.append(train_one(spec, seed, steps, eval_every, eval_episodes,
                                       final_episodes, output_dir, dqn_base=dqn_config))
    manifest = {
        "suite": args.suite, "seeds": seeds,
        "note": "This file lists runs launched in this invocation, not historical runs.",
        "runs": [item["run_id"] for item in summaries],
        "gymnasium_version": gymnasium.__version__,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "last_invocation.json").write_text(json.dumps(manifest, indent=2),
                                                     encoding="utf-8")
    print(f"All requested experiments finished. Output: {output_dir}")


if __name__ == "__main__":
    main()
