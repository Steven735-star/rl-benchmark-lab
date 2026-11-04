"""Finite-state discretization for CartPole observations."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class CartPoleDiscretizer:
    """Map the four continuous CartPole observations to a Q-table index.

    Clipping is essential because two velocity dimensions are unbounded in
    Gymnasium's observation space. The finite bounds here are an explicit
    experimental choice; users should examine their influence on performance.
    """

    bins: int
    lower: tuple[float, ...] = (-2.4, -3.0, -0.2095, -3.5)
    upper: tuple[float, ...] = (2.4, 3.0, 0.2095, 3.5)

    def __post_init__(self) -> None:
        if self.bins < 2 or len(self.lower) != 4 or len(self.upper) != 4:
            raise ValueError("Use at least two bins for all four CartPole dimensions.")
        if not np.all(np.asarray(self.upper) > np.asarray(self.lower)):
            raise ValueError("Each upper observation bound must exceed its lower bound.")

    @property
    def state_count(self) -> int:
        return self.bins ** 4

    def encode(self, observation: np.ndarray) -> int:
        """Clip outliers and encode the bin coordinates into one integer."""
        values = np.asarray(observation, dtype=np.float64)
        if values.shape != (4,) or not np.all(np.isfinite(values)):
            raise ValueError("CartPole observations must have four finite values.")
        coordinates = []
        for value, lower, upper in zip(values, self.lower, self.upper):
            boundaries = np.linspace(lower, upper, self.bins + 1)[1:-1]
            coordinates.append(int(np.searchsorted(boundaries, value, side="right")))
        return int(np.ravel_multi_index(tuple(coordinates), (self.bins,) * 4))
