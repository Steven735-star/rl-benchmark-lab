"""Deterministic evaluation independent of the training environment."""
import numpy as np


def evaluate_agent(agent, environment, episodes: int, seed: int) -> tuple[float, float]:
    """Return the mean and sample standard deviation of greedy returns."""
    returns = []
    for episode in range(episodes):
        observation, _ = environment.reset(seed=seed + episode)
        episode_return = 0.0
        while True:
            action = agent.greedy_act(observation)
            observation, reward, terminated, truncated, _ = environment.step(action)
            episode_return += float(reward)
            if terminated or truncated:
                break
        returns.append(episode_return)
    values = np.asarray(returns, dtype=np.float64)
    std = float(values.std(ddof=1)) if episodes > 1 else 0.0
    return float(values.mean()), std


def linear_epsilon(step: int, total_steps: int, start: float,
                   end: float, decay_fraction: float) -> float:
    """Decay exploration using environment steps rather than episode count."""
    decay_steps = max(1, round(total_steps * decay_fraction))
    fraction = min(1.0, max(0.0, step / decay_steps))
    return float(start + fraction * (end - start))
