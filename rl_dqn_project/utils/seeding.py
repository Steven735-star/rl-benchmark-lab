"""Explicit sources of reproducibility for CPU experiments."""
import random
import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Set Python, NumPy, and PyTorch random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # CPU is the intended baseline; deterministic PyTorch algorithms help
    # detect unexpected nondeterministic operators in that configuration.
    torch.use_deterministic_algorithms(True)
