"""Check Double DQN selection/evaluation and replay sample contract."""
import numpy as np
import pytest
import torch
from torch import nn
from agents.dqn import DQNAgent, DQNConfig, ReplayBuffer


class FixedQ(nn.Module):
    """A tiny known-Q network isolates the target computation from training."""
    def __init__(self, q_values):
        super().__init__()
        self.register_buffer("values", torch.tensor(q_values, dtype=torch.float32))

    def forward(self, observations):
        return self.values.unsqueeze(0).expand(observations.shape[0], -1)


@pytest.mark.parametrize("is_double,has_target,expected", [
    (False, True, 10.0),
    (True, True, 1.0),
    (False, False, 5.0),
])
def test_dqn_bellman_targets(is_double, has_target, expected):
    config = DQNConfig(gamma=1.0, double_dqn=is_double,
                       use_target_network=has_target)
    agent = DQNAgent(4, 2, config)
    agent.online = FixedQ([2.0, 5.0])
    agent.target = FixedQ([10.0, 1.0])
    next_states = torch.zeros((2, 4))
    rewards = torch.tensor([0.0, 7.0])
    terminated = torch.tensor([0.0, 1.0])
    target = agent._compute_targets(next_states, rewards, terminated)
    torch.testing.assert_close(target, torch.tensor([expected, 7.0]))


def test_replay_buffer_wrap_and_sample():
    buffer = ReplayBuffer(capacity=4, observation_count=3, seed=4)
    for index in range(7):
        state = np.full(3, index, dtype=np.float32)
        buffer.add(state, 1, float(index), state + 1, terminated=index % 2 == 0)
    assert buffer.size == 4
    assert buffer.position == 3
    states, actions, rewards, next_states, terminated = buffer.sample(4, torch.device('cpu'))
    assert states.shape == (4, 3)
    assert set(states[:, 0].numpy()).issubset({3, 4, 5, 6})
    assert actions.dtype == torch.int64
    assert terminated.shape == (4,)


def test_dqn_optimizer_and_checkpoint(tmp_path):
    torch.manual_seed(1)
    cfg = DQNConfig(learning_starts=8, batch_size=8, replay_capacity=50,
                    hidden_width=16, train_frequency=1, target_update_frequency=5)
    agent = DQNAgent(4, 2, cfg, seed=2)
    for i in range(12):
        state = np.full(4, i / 10, dtype=np.float32)
        agent.observe(state, i % 2, 1.0, state + 0.1, terminated=(i % 4 == 0))
        agent.learn(i + 1)
    assert agent.updates > 0
    assert np.isfinite(agent.last_loss)
    path = tmp_path / "dqn.pt"
    agent.save(str(path))
    loaded = DQNAgent.load(path)
    np.testing.assert_allclose(agent.online(torch.zeros(1, 4)).detach().numpy(),
                               loaded.online(torch.zeros(1, 4)).detach().numpy())


def test_invalid_double_dqn_without_target_network():
    with pytest.raises(ValueError):
        DQNConfig(double_dqn=True, use_target_network=False).validate()
