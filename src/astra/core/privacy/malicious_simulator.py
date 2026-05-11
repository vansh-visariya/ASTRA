"""
Malicious Client Simulator.

Implements various attack strategies for testing Byzantine robustness:
- Label flipping
- Additive noise
- Gradient sign flip
- Gradient scaling
- Targeted backdoor attacks

References:
- Biggio et al., "Poisoning Attacks against Support Vector Machines"
- Gu et al., "BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain"
"""

import hashlib
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch


class MaliciousSimulator:
    """Simulates malicious client behaviors for testing."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.malicious_config = config.get('malicious', {})
        
        self.behaviors = self.malicious_config.get('behaviors', ['noise'])
        
        self.logger = logging.getLogger(__name__)
    
    def inject_attack(
        self,
        gradient: np.ndarray,
        client_id: str
    ) -> np.ndarray:
        """
        Inject malicious behavior into gradient.
        
        Args:
            gradient: Original gradient vector.
            client_id: Client identifier for deterministic behavior.
        
        Returns:
            Modified gradient.
        """
        behavior = self._select_behavior(client_id)
        
        if behavior == 'noise':
            return self._add_noise(gradient)
        elif behavior == 'sign_flip':
            return self._sign_flip(gradient)
        elif behavior == 'scale':
            return self._scale_attack(gradient)
        elif behavior == 'label_flip':
            return gradient
        elif behavior == 'backdoor':
            return self._backdoor_attack(gradient, client_id)
        else:
            return gradient
    
    def _select_behavior(self, client_id: str) -> str:
        """Select attack behavior based on client ID."""
        hash_val = int(hashlib.md5(client_id.encode()).hexdigest(), 16)
        return self.behaviors[hash_val % len(self.behaviors)]
    
    def _add_noise(self, gradient: np.ndarray, noise_scale: float = 10.0) -> np.ndarray:
        """Add large random noise to gradient."""
        noise = np.random.randn(*gradient.shape) * noise_scale
        return gradient + noise
    
    def _sign_flip(self, gradient: np.ndarray) -> np.ndarray:
        """Flip sign of gradient."""
        return -gradient * 2.0
    
    def _scale_attack(self, gradient: np.ndarray, scale: float = 100.0) -> np.ndarray:
        """Scale gradient by large factor."""
        return gradient * scale
    
    def _backdoor_attack(self, gradient: np.ndarray, client_id: str) -> np.ndarray:
        """Inject backdoor pattern into gradient."""
        hash_val = int(hashlib.md5(client_id.encode()).hexdigest(), 16)
        
        backdoor_pattern = np.zeros_like(gradient)
        
        pattern_size = min(100, len(backdoor_pattern) // 100)
        for i in range(pattern_size):
            idx = (hash_val + i) % len(backdoor_pattern)
            backdoor_pattern[idx] = 1.0
        
        return gradient + backdoor_pattern * 0.5
    
    def simulate_label_flip(
        self,
        labels: torch.Tensor,
        flip_ratio: float = 0.5
    ) -> torch.Tensor:
        """Simulate label flipping attack on data."""
        num_samples = len(labels)
        num_flip = int(num_samples * flip_ratio)
        
        indices = torch.randperm(num_samples)[:num_flip]
        
        flipped_labels = labels.clone()
        flipped_labels[indices] = (labels[indices] + 1) % 10
        
        return flipped_labels
    
    def compute_attack_impact(
        self,
        original_accuracy: float,
        attacked_accuracy: float
    ) -> Dict[str, float]:
        """
        Compute metrics for attack impact.
        
        Args:
            original_accuracy: Accuracy before attack.
            attacked_accuracy: Accuracy after attack.
        
        Returns:
            Dictionary with impact metrics.
        """
        accuracy_drop = original_accuracy - attacked_accuracy
        relative_drop = accuracy_drop / max(original_accuracy, 1e-8)
        
        return {
            'accuracy_drop': accuracy_drop,
            'relative_drop': relative_drop,
            'attack_success': 1.0 if relative_drop > 0.1 else 0.0
        }
    
    def get_suspicion_score(self, gradient: np.ndarray) -> float:
        """
        Compute suspicion score for a gradient.
        
        Higher scores indicate more suspicious patterns.
        
        Args:
            gradient: Client gradient.
        
        Returns:
            Suspicion score between 0 and 1.
        """
        norm = np.linalg.norm(gradient)
        mean_abs = np.mean(np.abs(gradient))
        
        score = 0.0
        
        if norm > 10.0:
            score += 0.3
        
        if mean_abs > 1.0:
            score += 0.2
        
        sign_consistency = np.mean(np.sign(gradient) ** 2)
        if sign_consistency > 0.9:
            score += 0.2
        
        return min(score, 1.0)
