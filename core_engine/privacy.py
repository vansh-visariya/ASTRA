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

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def clip_and_noise(
    gradient: np.ndarray,
    clip_norm: float,
    sigma: float
) -> np.ndarray:
    """
    Apply DP clipping and add Gaussian noise.
    
    Args:
        gradient: Input gradient vector.
        clip_norm: L2 norm clipping threshold.
        sigma: Gaussian noise standard deviation.
    
    Returns:
        Clipped and noised gradient.
    """
    grad_norm = np.linalg.norm(gradient)
    
    if grad_norm > clip_norm:
        gradient = gradient * (clip_norm / grad_norm)
    
    noise = np.random.normal(0, sigma, gradient.shape)
    
    return gradient + noise


def secure_aggregate_masking(
    updates: List[np.ndarray],
    seed: int
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
    """
    np.random.seed(seed)
    
    masks = []
    for update in updates:
        mask = np.random.randn(*update.shape)
        masks.append(mask)
    
    masked_updates = [u + m for u, m in zip(updates, masks)]
    
    aggregated = sum(masked_updates)
    
    return aggregated


def estimate_epsilon(
    steps: int,
    sigma: float,
    clip_norm: float,
    delta: float = 1e-5
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
    """
    c = clip_norm / sigma
    
    sigma_squared = sigma ** 2
    
    epsilon_per_step = math.sqrt(2 * math.log(1.25 / delta)) * c
    
    total_epsilon = epsilon_per_step * math.sqrt(steps)
    
    return total_epsilon


class MomentsAccountant:
    """Simple moments accountant for tracking DP budget."""
    
    def __init__(self, sigma: float, clip_norm: float, delta: float = 1e-5):
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
        q = 1.0
        
        alpha = 1.0 + self.noise_scale ** 2
        
        epsilon_step = (alpha - 1) + math.sqrt(
            2 * alpha * math.log(1.25 * math.sqrt(self.steps) / self.delta)
        ) * self.noise_scale
        
        self.epsilon = math.sqrt(self.steps) * epsilon_step
    
    def get_epsilon(self) -> float:
        """Get current epsilon estimate."""
        return self.epsilon
    
    def get_epsilon_delta(self) -> Tuple[float, float]:
        """Get current epsilon and delta."""
        return self.epsilon, self.delta


def apply_dp_to_gradient(
    gradient: np.ndarray,
    config: Dict[str, Any]
) -> np.ndarray:
    """Apply DP preprocessing to gradient."""
    if not config.get('privacy', {}).get('dp_enabled', False):
        return gradient
    
    clip_norm = config['privacy'].get('clip_norm', 1.0)
    sigma = config['privacy'].get('sigma', 1.0)
    
    return clip_and_noise(gradient, clip_norm, sigma)


def create_secure_aggregation(
    updates: List[np.ndarray],
    config: Dict[str, Any]
) -> np.ndarray:
    """Create secure aggregation of updates."""
    seed = config.get('experiment_id', 'default')
    
    if isinstance(seed, str):
        seed = hash(seed) % (2 ** 31)
    
    return secure_aggregate_masking(updates, seed)
