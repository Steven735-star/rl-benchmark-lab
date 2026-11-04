"""Independent DQN, Double DQN, target ablation, and experience replay."""
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
import torch
from torch import nn


@dataclass
class DQNConfig:
    """Hyperparameters stored alongside each trained model."""
    gamma: float = 0.99
    learning_rate: float = 5e-4
    hidden_width: int = 128
    batch_size: int = 64
    replay_capacity: int = 50_000
    learning_starts: int = 1_000
    train_frequency: int = 4
    target_update_frequency: int = 1_000
    gradient_clip: float = 10.0
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_fraction: float = 0.5
    double_dqn: bool = False
    use_target_network: bool = True

    def validate(self) -> None:
        if not (0 < self.gamma <= 1) or self.learning_rate <= 0:
            raise ValueError("Invalid gamma or learning rate.")
        if self.hidden_width <= 0 or self.batch_size <= 0:
            raise ValueError("Invalid neural network or batch size.")
        if self.replay_capacity < self.batch_size:
            raise ValueError("Replay capacity must be at least the batch size.")
        if self.learning_starts < 0 or self.train_frequency <= 0:
            raise ValueError("Invalid learning-start threshold or train frequency.")
        if self.target_update_frequency <= 0 or not (0 < self.epsilon_decay_fraction <= 1):
            raise ValueError("Invalid target synchronization or exploration schedule.")
        if not (0 <= self.epsilon_end <= self.epsilon_start <= 1):
            raise ValueError("Expected 0 <= epsilon_end <= epsilon_start <= 1.")
        if self.double_dqn and not self.use_target_network:
            raise ValueError("Double DQN requires a separate target network.")


class QNetwork(nn.Module):
    """Feed-forward network returning one action value per discrete action."""
    def __init__(self, observation_count: int, action_count: int, hidden_width: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(observation_count, hidden_width), nn.ReLU(),
            nn.Linear(hidden_width, hidden_width), nn.ReLU(),
            nn.Linear(hidden_width, action_count),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.layers(observations)


class ReplayBuffer:
    """Preallocated ring buffer with reproducible random minibatches."""
    def __init__(self, capacity: int, observation_count: int, seed: int) -> None:
        self.capacity = capacity
        self.states = np.empty((capacity, observation_count), dtype=np.float32)
        self.next_states = np.empty_like(self.states)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.terminated = np.empty(capacity, dtype=np.float32)
        self.size = 0
        self.position = 0
        self.rng = np.random.default_rng(seed)

    def add(self, state: np.ndarray, action: int, reward: float,
            next_state: np.ndarray, terminated: bool) -> None:
        index = self.position
        self.states[index] = state
        self.actions[index] = action
        self.rewards[index] = reward
        self.next_states[index] = next_state
        self.terminated[index] = float(terminated)
        self.position = (index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, ...]:
        if self.size < batch_size:
            raise ValueError("Not enough transitions in replay buffer.")
        indices = self.rng.integers(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.states[indices], device=device),
            torch.as_tensor(self.actions[indices], device=device),
            torch.as_tensor(self.rewards[indices], device=device),
            torch.as_tensor(self.next_states[indices], device=device),
            torch.as_tensor(self.terminated[indices], device=device),
        )


class DQNAgent:
    """DQN with switchable Double DQN and no-target-network ablation."""
    def __init__(self, observation_count: int, action_count: int,
                 config: DQNConfig, seed: int = 0, device: str = "cpu") -> None:
        config.validate()
        self.config = config
        self.observation_count = observation_count
        self.action_count = action_count
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed)
        self.online = QNetwork(observation_count, action_count, config.hidden_width).to(self.device)
        self.target = deepcopy(self.online).to(self.device)
        self.target.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config.learning_rate)
        self.buffer = ReplayBuffer(config.replay_capacity, observation_count, seed + 12345)
        self.last_loss: float | None = None
        self.updates = 0

    def act(self, observation: np.ndarray, epsilon: float = 0.0) -> int:
        """Greedy evaluation uses epsilon=0; training uses a decaying epsilon."""
        if epsilon > 0.0 and self.rng.random() < epsilon:
            return int(self.rng.integers(self.action_count))
        with torch.no_grad():
            inputs = torch.as_tensor(np.asarray(observation, dtype=np.float32),
                                      device=self.device).unsqueeze(0)
            return int(self.online(inputs).argmax(dim=1).item())

    def greedy_act(self, observation: np.ndarray) -> int:
        """Evaluate greedily without sampling the exploration RNG."""
        return self.act(observation, epsilon=0.0)

    def observe(self, state: np.ndarray, action: int, reward: float,
                next_state: np.ndarray, terminated: bool) -> None:
        self.buffer.add(state, action, reward, next_state, terminated)

    def _compute_targets(self, next_states: torch.Tensor, rewards: torch.Tensor,
                         terminated: torch.Tensor) -> torch.Tensor:
        """Select/evaluate next actions according to the experimental variant."""
        with torch.no_grad():
            if not self.config.use_target_network:
                next_values = self.online(next_states).max(dim=1).values
            elif self.config.double_dqn:
                next_actions = self.online(next_states).argmax(dim=1, keepdim=True)
                next_values = self.target(next_states).gather(1, next_actions).squeeze(1)
            else:
                next_values = self.target(next_states).max(dim=1).values
            return rewards + self.config.gamma * (1.0 - terminated) * next_values

    def learn(self, environment_step: int) -> float | None:
        """Perform one gradient step when warmup and train frequency permit."""
        c = self.config
        loss_value = None
        if (environment_step >= c.learning_starts
                and environment_step % c.train_frequency == 0
                and self.buffer.size >= c.batch_size):
            states, actions, rewards, next_states, terminated = self.buffer.sample(
                c.batch_size, self.device)
            predicted = self.online(states).gather(1, actions[:, None]).squeeze(1)
            target = self._compute_targets(next_states, rewards, terminated)
            loss = nn.functional.smooth_l1_loss(predicted, target)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.online.parameters(), c.gradient_clip)
            self.optimizer.step()
            self.updates += 1
            self.last_loss = float(loss.item())
            loss_value = self.last_loss
        if c.use_target_network and environment_step % c.target_update_frequency == 0:
            self.target.load_state_dict(self.online.state_dict())
        return loss_value

    def save(self, filepath: str) -> None:
        """Persist model parameters and all data required to reconstruct it."""
        torch.save({
            "online_weights": self.online.state_dict(),
            "config": asdict(self.config),
            "observation_count": self.observation_count,
            "action_count": self.action_count,
        }, filepath)

    @classmethod
    def load(cls, filepath: str | Path, device: str = "cpu") -> "DQNAgent":
        # weights_only avoids loading arbitrary Python objects from checkpoints.
        checkpoint = torch.load(filepath, map_location=device, weights_only=True)
        agent = cls(checkpoint["observation_count"], checkpoint["action_count"],
                    DQNConfig(**checkpoint["config"]), device=device)
        agent.online.load_state_dict(checkpoint["online_weights"])
        agent.target.load_state_dict(agent.online.state_dict())
        return agent
