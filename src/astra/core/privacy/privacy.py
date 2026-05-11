"""
Privacy Mechanisms for Federated Learning.

Implements:
- Differential Privacy (DP-SGD style): gradient clipping + Gaussian noise
- Secure Aggregation simulation (additive masking protocol)
- Epsilon estimation using moments accountant

References:
- Abadi et al., "Deep Learning with Differential Privacy"
- Bonawitz et al., "Practical Secure Aggregation for Privacy-Preserving Machine Learning"
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.random import default_rng

__all__ = [
    "clip_and_noise",
    "secure_aggregate_masking",
    "estimate_epsilon",
    "MomentsAccountant",
    "apply_dp_to_gradient",
    "create_secure_aggregation",
]


def clip_and_noise(
    gradient: np.ndarray,
    clip_norm: float,
    sigma: float,
) -> np.ndarray:
    """
    Apply DP clipping and add Gaussian noise.

    Args:
        gradient: Input gradient vector.
        clip_norm: L2 norm clipping threshold.
        sigma: Gaussian noise standard deviation.

    Returns:
        Clipped and noised gradient.

    Raises:
        ValueError: If clip_norm or sigma is invalid.
    """
    if clip_norm <= 0:
        raise ValueError("clip_norm must be positive")
    if sigma < 0:
        raise ValueError("sigma must be non-negative")

    grad_norm = np.linalg.norm(gradient)

    if grad_norm > clip_norm:
        gradient = gradient * (clip_norm / grad_norm)

    rng = default_rng()
    noise = rng.standard_normal(gradient.shape) * sigma

    return gradient + noise


def secure_aggregate_masking(
    updates: list[np.ndarray],
    seed: int,
) -> np.ndarray:
    """
    Simulate secure aggregation using additive masking.

    In real implementation, this would use cryptographic protocols
    to ensure server cannot see individual updates.

    Args:
        updates: List of client update vectors.
        seed: Random seed for reproducible masking.

    Returns:
        Aggregated result (sum of masked updates).

    Raises:
        ValueError: If updates list is empty.
    """
    if not updates:
        raise ValueError("Cannot aggregate empty updates list")

    rng = default_rng(seed)

    masks = [rng.standard_normal(update.shape) for update in updates]
    masked_updates = [u + m for u, m in zip(updates, masks)]

    return sum(masked_updates)


def estimate_epsilon(
    steps: int,
    sigma: float,
    clip_norm: float,
    delta: float = 1e-5,
) -> float:
    """
    Estimate privacy budget (epsilon) using simple composition.

    This is a simplified estimate. For formal guarantees, use
    moments accountant or Rényi DP.

    Args:
        steps: Number of training steps.
        sigma: Noise multiplier.
        clip_norm: Clipping norm.
        delta: Target delta for (epsilon, delta)-DP.

    Returns:
        Estimated epsilon value.

    Raises:
        ValueError: If parameters are invalid.
    """
    if steps <= 0:
        raise ValueError("steps must be positive")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if clip_norm <= 0:
        raise ValueError("clip_norm must be positive")
    if not 0 < delta < 1:
        raise ValueError("delta must be between 0 and 1")

    c = clip_norm / sigma
    epsilon_per_step = math.sqrt(2 * math.log(1.25 / delta)) * c

    return epsilon_per_step * math.sqrt(steps)


class MomentsAccountant:
    """Simple moments accountant for tracking DP budget."""

    def __init__(
        self,
        sigma: float,
        clip_norm: float,
        delta: float = 1e-5,
    ) -> None:
        """
        Initialize moments accountant.

        Args:
            sigma: Noise multiplier.
            clip_norm: L2 norm clipping threshold.
            delta: Target delta for (epsilon, delta)-DP.

        Raises:
            ValueError: If parameters are invalid.
        """
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        if clip_norm <= 0:
            raise ValueError("clip_norm must be positive")
        if not 0 < delta < 1:
            raise ValueError("delta must be between 0 and 1")

        self.sigma = sigma
        self.clip_norm = clip_norm
        self.delta = delta
        self.steps = 0
        self.epsilon = 0.0
        self.noise_scale = clip_norm / sigma

    def step(self) -> None:
        """Record one training step."""
        self.steps += 1
        self._update_epsilon()

    def _update_epsilon(self) -> None:
        """Update epsilon estimate using strong composition."""
        alpha = 1.0 + self.noise_scale**2

        epsilon_step = (alpha - 1) + math.sqrt(
            2 * alpha * math.log(1.25 * math.sqrt(self.steps) / self.delta)
        ) * self.noise_scale

        self.epsilon = math.sqrt(self.steps) * epsilon_step

    def get_epsilon(self) -> float:
        """Get current epsilon estimate."""
        return self.epsilon

    def get_epsilon_delta(self) -> tuple[float, float]:
        """Get current epsilon and delta."""
        return self.epsilon, self.delta


def apply_dp_to_gradient(
    gradient: np.ndarray,
    config: dict[str, Any],
) -> np.ndarray:
    """
    Apply DP preprocessing to gradient.

    Args:
        gradient: Input gradient.
        config: Configuration dictionary with privacy settings.

    Returns:
        Processed gradient.
    """
    privacy_config = config.get("privacy", {})
    if not privacy_config.get("dp_enabled", False):
        return gradient

    clip_norm = privacy_config.get("clip_norm", 1.0)
    sigma = privacy_config.get("sigma", 1.0)

    return clip_and_noise(gradient, clip_norm, sigma)


def create_secure_aggregation(
    updates: list[np.ndarray],
    config: dict[str, Any],
) -> np.ndarray:
    """
    Create secure aggregation of updates.

    Args:
        updates: List of client updates.
        config: Configuration dictionary.

    Returns:
        Securely aggregated result.
    """
    seed = config.get("experiment_id", "default")

    if isinstance(seed, str):
        seed = hash(seed) % (2**31)

    return secure_aggregate_masking(updates, seed)
