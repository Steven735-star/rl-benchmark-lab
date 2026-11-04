"""Tabular Q-learning implemented with NumPy, not a ready-made RL agent."""
import numpy as np
from envs.discretizer import CartPoleDiscretizer


class TabularQLearning:
    """An epsilon-greedy agent using a discrete state-action value table."""

    def __init__(self, bins: int, action_count: int, learning_rate: float = 0.1,
                 gamma: float = 0.99, seed: int = 0) -> None:
        if action_count < 2:
            raise ValueError("action_count must be at least two.")
        self.discretizer = CartPoleDiscretizer(bins)
        self.q_table = np.zeros((self.discretizer.state_count, action_count), dtype=np.float32)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.rng = np.random.default_rng(seed)
        self.action_count = action_count

    def act(self, observation: np.ndarray, epsilon: float = 0.0) -> int:
        """Select uniformly among tied greedy actions to avoid action bias."""
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.action_count))
        q_values = self.q_table[self.discretizer.encode(observation)]
        best_actions = np.flatnonzero(q_values == q_values.max())
        return int(self.rng.choice(best_actions))

    def greedy_act(self, observation: np.ndarray) -> int:
        """Evaluate without changing the training random-number stream."""
        return int(self.q_table[self.discretizer.encode(observation)].argmax())

    def update(self, observation: np.ndarray, action: int, reward: float,
               next_observation: np.ndarray, terminated: bool) -> float:
        """Use terminated, not truncated, to decide whether to bootstrap."""
        state_index = self.discretizer.encode(observation)
        next_index = self.discretizer.encode(next_observation)
        next_value = 0.0 if terminated else float(self.q_table[next_index].max())
        target = reward + self.gamma * next_value
        error = target - float(self.q_table[state_index, action])
        self.q_table[state_index, action] += self.learning_rate * error
        return error

    def save(self, filepath: str) -> None:
        """Store the Q-table and configuration without using pickle."""
        np.savez_compressed(filepath, q_table=self.q_table,
                            bins=self.discretizer.bins,
                            learning_rate=self.learning_rate, gamma=self.gamma)

    @classmethod
    def load(cls, filepath: str, seed: int = 0) -> "TabularQLearning":
        with np.load(filepath, allow_pickle=False) as data:
            q_table = data["q_table"].copy()
            agent = cls(int(data["bins"]), q_table.shape[1],
                        float(data["learning_rate"]), float(data["gamma"]), seed)
            agent.q_table[:] = q_table
        return agent
