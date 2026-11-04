"""Shared environment-step training and consistent, independent evaluation."""
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
import csv
import json
import platform

import numpy as np
import torch

from agents.tabular import TabularQLearning
from agents.dqn import DQNAgent, DQNConfig
from experiments.configs import ExperimentSpec
from utils.evaluation import evaluate_agent, linear_epsilon
from utils.seeding import seed_everything


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    """Write a header even when no episode has ended yet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def train_one(spec: ExperimentSpec, seed: int, total_steps: int, eval_every: int,
              eval_episodes: int, final_eval_episodes: int, results_dir: str | Path,
              tabular_learning_rate: float = 0.1, tabular_gamma: float = 0.99,
              dqn_base: DQNConfig | None = None, environment_factory=None) -> dict:
    """Train one seed, then write raw data, final evaluation, and model weights.

    Environment factory injection enables testing with a small artificial MDP.
    Otherwise the function uses the official Gymnasium environment directly.
    """
    if min(total_steps, eval_every, eval_episodes, final_eval_episodes) <= 0:
        raise ValueError("Training and evaluation counts must be positive.")
    if environment_factory is None:
        import gymnasium as gym
        environment_factory = gym.make
    seed_everything(seed)
    train_env = environment_factory(spec.env_id)
    eval_env = environment_factory(spec.env_id)
    if not hasattr(train_env.action_space, "n") or len(train_env.observation_space.shape) != 1:
        raise ValueError("This implementation requires vector observations and discrete actions.")
    action_count = int(train_env.action_space.n)
    observation_count = int(np.prod(train_env.observation_space.shape))
    if spec.algorithm == "tabular":
        if spec.env_id != "CartPole-v1" or spec.bins is None:
            raise ValueError("Tabular state discretization is implemented for CartPole only.")
        agent = TabularQLearning(spec.bins, action_count, tabular_learning_rate,
                                 tabular_gamma, seed)
        config = {"bins": spec.bins, "learning_rate": tabular_learning_rate,
                  "gamma": tabular_gamma, "epsilon_start": 1.0,
                  "epsilon_end": 0.05, "epsilon_decay_fraction": 0.8}
        extension = "npz"
    elif spec.algorithm == "dqn":
        base_dict = asdict(dqn_base or DQNConfig())
        base_dict.update(spec.overrides)
        config = base_dict
        agent = DQNAgent(observation_count, action_count, DQNConfig(**base_dict), seed)
        extension = "pt"
    else:
        raise ValueError(f"Unsupported algorithm: {spec.algorithm}")

    out = Path(results_dir)
    metrics_dir, models_dir = out / "metrics", out / "models"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"{spec.name}_seed{seed}"
    evaluations: list[dict] = []
    episodes: list[dict] = []
    evaluation_time = 0.0
    started = perf_counter()
    observation, _ = train_env.reset(seed=seed)
    train_env.action_space.seed(seed)
    current_return = 0.0
    current_length = 0
    episode_number = 0
    provisional_threshold = None
    threshold = 475 if spec.env_id == "CartPole-v1" else -100

    def record_evaluation(step: int, episode_count: int, kind: str) -> tuple[float, float]:
        nonlocal evaluation_time, provisional_threshold
        evaluation_started = perf_counter()
        # Common random numbers make runs within one seed more comparable.
        evaluation_seed = (7_000_000 + seed * 1_000 if kind == "final"
                           else 1_000_000 + seed * 100_000 + (step // eval_every) * 1_000)
        mean, std = evaluate_agent(agent, eval_env, episode_count, evaluation_seed)
        evaluation_time += perf_counter() - evaluation_started
        evaluations.append({"env_steps": step, "kind": kind, "mean_return": mean,
                            "std_return": std, "evaluation_episodes": episode_count})
        # Intermediate 10-episode evaluations are provisional, not a verified
        # 100-episode success threshold.
        if kind == "periodic" and mean >= threshold and provisional_threshold is None:
            provisional_threshold = step
        return mean, std

    try:
        record_evaluation(0, eval_episodes, "periodic")
        for global_step in range(1, total_steps + 1):
            epsilon = linear_epsilon(global_step - 1, total_steps, config["epsilon_start"],
                                     config["epsilon_end"], config["epsilon_decay_fraction"])
            action = agent.act(observation, epsilon)
            next_observation, reward, terminated, truncated, _ = train_env.step(action)
            # Truncation ends rollout, but does not indicate an MDP terminal
            # state and must not disable bootstrapping in the Bellman target.
            if spec.algorithm == "tabular":
                agent.update(observation, action, float(reward), next_observation, terminated)
            else:
                agent.observe(observation, action, float(reward), next_observation, terminated)
                agent.learn(global_step)
            observation = next_observation
            current_return += float(reward)
            current_length += 1

            if terminated or truncated:
                episode_number += 1
                episodes.append({"env_steps": global_step, "episode": episode_number,
                                 "train_return": current_return, "length": current_length,
                                 "epsilon": epsilon})
                observation, _ = train_env.reset(seed=seed + episode_number)
                current_return, current_length = 0.0, 0

            if global_step % eval_every == 0 or global_step == total_steps:
                mean, _ = record_evaluation(global_step, eval_episodes, "periodic")
                print(f"[{run_id}] steps={global_step}/{total_steps}, greedy_return={mean:.1f}",
                      flush=True)

        final_mean, final_std = record_evaluation(total_steps, final_eval_episodes, "final")
        wall_clock_total = perf_counter() - started
        checkpoint_path = models_dir / f"{run_id}.{extension}"
        agent.save(str(checkpoint_path))
        evaluation_columns = ["env_steps", "kind", "mean_return", "std_return",
                              "evaluation_episodes"]
        episode_columns = ["env_steps", "episode", "train_return", "length", "epsilon"]
        write_csv(metrics_dir / f"{run_id}_evaluations.csv", evaluations, evaluation_columns)
        write_csv(metrics_dir / f"{run_id}_episodes.csv", episodes, episode_columns)

        summary = {
            "run_id": run_id, "experiment": spec.name, "suite": spec.suite,
            "env_id": spec.env_id, "algorithm": spec.algorithm, "seed": seed,
            "environment_steps": total_steps, "evaluation_interval": eval_every,
            "periodic_evaluation_episodes": eval_episodes,
            "final_evaluation_episodes": final_eval_episodes,
            "final_mean_return": final_mean, "final_std_return": final_std,
            "success_threshold": threshold,
            "reached_threshold_final_100episodes": (final_mean >= threshold
                    if final_eval_episodes == 100 else None),
            "first_provisional_10episode_threshold_step": provisional_threshold,
            "wall_clock_seconds_total": wall_clock_total,
            "wall_clock_seconds_excluding_evaluations": wall_clock_total - evaluation_time,
            "evaluation_seconds": evaluation_time,
            "checkpoint": str(checkpoint_path.relative_to(out)),
            "hyperparameters": config,
            "torch_version": torch.__version__,
            "python_version": platform.python_version(),
            "cpu_description": platform.processor() or platform.machine(),
            "device": "cpu", "neural_network_updates": getattr(agent, "updates", None),
        }
        with (metrics_dir / f"{run_id}_summary.json").open("w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2, ensure_ascii=False)
        print(f"Completed {run_id}: final greedy mean={final_mean:.2f}", flush=True)
        return summary
    finally:
        train_env.close()
        eval_env.close()
