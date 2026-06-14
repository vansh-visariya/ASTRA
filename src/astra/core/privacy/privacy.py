"""
Privacy Mechanisms for Federated Learning.

Implements Differential Privacy (DP-SGD style): gradient clipping + Gaussian noise.
"""

from __future__ import annotations

import numpy as np
from numpy.random import default_rng


def clip_and_noise(
    gradient: np.ndarray,
    clip_norm: float,
    sigma: float = 1.0,
) -> np.ndarray:
    """
    Apply DP-SGD: clip gradient norm then add Gaussian noise.

    Args:
        gradient: Input gradient vector.
        clip_norm: L2 norm clipping threshold.
        sigma: Standard deviation of Gaussian noise.

    Returns:
        Noisy, clipped gradient.
    """
    norm = float(np.linalg.norm(gradient))
    scale = min(1.0, clip_norm / (norm + 1e-8))

    gradient = gradient * scale

    rng = default_rng()
    noise = rng.standard_normal(gradient.shape) * sigma

    return gradient + noise


__all__ = ["clip_and_noise"]
