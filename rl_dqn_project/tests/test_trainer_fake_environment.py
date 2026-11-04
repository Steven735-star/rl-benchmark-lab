"""End-to-end pipeline tests without requiring an installed Gymnasium package."""
import csv
from pathlib import Path
from types import SimpleNamespace

from agents.dqn import DQNConfig
from experiments.configs import ExperimentSpec, get_experiments
from experiments.trainer import train_one
from scripts.plot_results import load_records, plot_evaluations, write_summary
import numpy as np


class FakeEnvironment:
    """Simple finite-horizon stand-in to test data flow, not RL performance."""
    def __init__(self, env_id):
        self.env_id = env_id
        self.dim = 4 if env_id == "CartPole-v1" else 6
        self.action_space = SimpleNamespace(n=2 if self.dim == 4 else 3,
                                            seed=lambda seed: None)
        self.observation_space = SimpleNamespace(shape=(self.dim,))
        self.step_count = 0

    def reset(self, seed=None):
        self.step_count = 0
        return np.full(self.dim, 0.01, dtype=np.float32), {}

    def step(self, action):
        self.step_count += 1
        state = np.full(self.dim, (self.step_count % 4) / 100, dtype=np.float32)
        # Alternate termination and truncation for episode-boundary coverage.
        terminated = self.step_count == 3 and action == 0
        truncated = self.step_count >= 6
        return state, float(action == 0), terminated, truncated, {}

    def close(self):
        pass


def test_experiment_design_counts():
    assert len(get_experiments("tier1")) == 4
    assert len(get_experiments("tier2")) == 6
    assert len(get_experiments("tier3")) == 1


def test_tabular_and_double_dqn_training_pipeline(tmp_path):
    specs = [get_experiments("tier1")[0], get_experiments("tier3")[0]]
    for spec in specs:
        run = train_one(spec, seed=0, total_steps=24, eval_every=12,
                        eval_episodes=2, final_eval_episodes=3, results_dir=tmp_path,
                        dqn_base=DQNConfig(learning_starts=4, batch_size=4,
                                           hidden_width=16, replay_capacity=30,
                                           train_frequency=1, target_update_frequency=5),
                        environment_factory=FakeEnvironment)
        assert run["environment_steps"] == 24
        assert run["reached_threshold_final_100episodes"] is None
        assert (tmp_path / "models" / run["checkpoint"].split('/')[-1]).exists()
        eval_path = tmp_path / "metrics" / f"{run['run_id']}_evaluations.csv"
        with eval_path.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        assert [int(r["env_steps"]) for r in rows] == [0, 12, 24, 24]
        assert rows[-1]["kind"] == "final"

    grouped = load_records(tmp_path, "all")
    figures = tmp_path / "figures"
    figures.mkdir()
    plot_evaluations(grouped, figures, "test")
    write_summary(grouped, figures, "test")
    assert (figures / "test_evaluation_curves.png").exists()
    assert (figures / "test_aggregate.csv").exists()
