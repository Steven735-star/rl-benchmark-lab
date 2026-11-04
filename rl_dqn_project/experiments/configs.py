"""Centralized experiment design: one-factor-at-a-time comparisons."""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    suite: str
    env_id: str
    algorithm: str
    bins: int | None = None
    overrides: dict = field(default_factory=dict)


def get_experiments(suite: str) -> list[ExperimentSpec]:
    """Return specified experiments, without double-counting shared baselines."""
    tier1 = [
        ExperimentSpec(f"cartpole_tabular_{b}bins", "tier1", "CartPole-v1", "tabular", bins=b)
        for b in (6, 10, 20)
    ] + [ExperimentSpec("cartpole_dqn", "tier1", "CartPole-v1", "dqn")]
    tier2 = [
        ExperimentSpec("acrobot_dqn_default", "tier2", "Acrobot-v1", "dqn"),
        ExperimentSpec("acrobot_lr_0p0001", "tier2", "Acrobot-v1", "dqn",
                       overrides={"learning_rate": 1e-4}),
        ExperimentSpec("acrobot_lr_0p001", "tier2", "Acrobot-v1", "dqn",
                       overrides={"learning_rate": 1e-3}),
        ExperimentSpec("acrobot_eps_0p2", "tier2", "Acrobot-v1", "dqn",
                       overrides={"epsilon_decay_fraction": 0.2}),
        ExperimentSpec("acrobot_eps_0p8", "tier2", "Acrobot-v1", "dqn",
                       overrides={"epsilon_decay_fraction": 0.8}),
        ExperimentSpec("acrobot_no_target", "tier2", "Acrobot-v1", "dqn",
                       overrides={"use_target_network": False}),
    ]
    tier3 = [
        ExperimentSpec("acrobot_double_dqn", "tier3", "Acrobot-v1", "dqn",
                       overrides={"double_dqn": True})
    ]
    experiments = tier1 + tier2 + tier3
    if suite == "all":
        return experiments
    if suite in {"tier1", "tier2", "tier3"}:
        return [spec for spec in experiments if spec.suite == suite]
    raise ValueError(f"Unknown suite: {suite}")
