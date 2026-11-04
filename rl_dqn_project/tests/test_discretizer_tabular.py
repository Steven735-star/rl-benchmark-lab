"""Unit tests for finite observation encoding and tabular Bellman updates."""
import numpy as np
import pytest
from envs.discretizer import CartPoleDiscretizer
from agents.tabular import TabularQLearning
from utils.evaluation import linear_epsilon


@pytest.mark.parametrize("bins", [6, 10, 20])
def test_discretizer_bounds_and_cardinality(bins):
    encoder = CartPoleDiscretizer(bins)
    assert encoder.state_count == bins ** 4
    assert encoder.encode(np.array([-100, -100, -100, -100])) == 0
    assert encoder.encode(np.array([100, 100, 100, 100])) == bins ** 4 - 1
    assert 0 <= encoder.encode(np.zeros(4)) < bins ** 4


def test_invalid_discretizer_input():
    with pytest.raises(ValueError):
        CartPoleDiscretizer(1)
    with pytest.raises(ValueError):
        CartPoleDiscretizer(6).encode(np.array([float('nan'), 0, 0, 0]))


def test_tabular_terminated_disables_bootstrap_but_truncation_does_not():
    agent = TabularQLearning(6, 2, learning_rate=1.0, gamma=0.9)
    source = np.array([-1.0, 0, 0, 0])
    successor = np.array([1.0, 0, 0, 0])
    succ_id = agent.discretizer.encode(successor)
    src_id = agent.discretizer.encode(source)
    agent.q_table[succ_id, 1] = 10
    agent.update(source, 0, 2, successor, terminated=False)
    assert agent.q_table[src_id, 0] == pytest.approx(11.0)
    agent.update(source, 0, 2, successor, terminated=True)
    assert agent.q_table[src_id, 0] == pytest.approx(2.0)


def test_linear_epsilon_decays_by_steps():
    assert linear_epsilon(0, 100, 1.0, 0.05, 0.5) == pytest.approx(1.0)
    assert linear_epsilon(25, 100, 1.0, 0.05, 0.5) == pytest.approx(0.525)
    assert linear_epsilon(100, 100, 1.0, 0.05, 0.5) == pytest.approx(0.05)


def test_greedy_evaluation_does_not_advance_training_rng():
    agent_a = TabularQLearning(6, 2, seed=5)
    agent_b = TabularQLearning(6, 2, seed=5)
    state = np.zeros(4)
    for _ in range(10):
        agent_a.greedy_act(state)
    assert [agent_a.act(state, epsilon=0.5) for _ in range(20)] == [
        agent_b.act(state, epsilon=0.5) for _ in range(20)]


def test_tabular_checkpoint_roundtrip(tmp_path):
    agent = TabularQLearning(6, 2, seed=5)
    agent.q_table[2, 1] = 13
    path = tmp_path / "agent.npz"
    agent.save(str(path))
    loaded = TabularQLearning.load(str(path))
    np.testing.assert_allclose(loaded.q_table, agent.q_table)
