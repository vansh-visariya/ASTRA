"""
Trust Manager for Dynamic Client Trust Scoring.

Implements trust scoring with soft quarantine for suspicious clients.

References:
- Cao et al., "Distributed Learning with Dynamic Trust Management"
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import numpy as np

__all__ = ["TrustManager"]


class TrustManager:
    """Manages dynamic trust scores for federated learning clients."""

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize TrustManager.

        Args:
            config: Configuration dictionary with trust settings.
        """
        if not isinstance(config, dict):
            raise ValueError("Config must be a dictionary")

        self.config = config
        self.trust_config = config.get("trust", {})

        if not isinstance(self.trust_config, dict):
            self.trust_config = {}

        self.init_trust = self.trust_config.get("init", 1.0)
        self.update_alpha = self.trust_config.get("update_alpha", 0.3)
        self.quarantine_threshold = self.trust_config.get("quarantine_threshold", 0.35)
        self.soft_decay = self.trust_config.get("soft_decay", 0.8)

        self.trust_scores: dict[str, float] = defaultdict(lambda: self.init_trust)
        self.quarantined: dict[str, bool] = defaultdict(bool)
        self.trust_history: dict[str, list[float]] = defaultdict(list)

        self.logger = logging.getLogger(__name__)

    def update_trust(
        self,
        client_id: str,
        update_vector: np.ndarray | None,
        global_vector: np.ndarray | None,
    ) -> float:
        """
        Update trust score for a client based on update similarity.

        Args:
            client_id: Client identifier.
            update_vector: Client's update vector.
            global_vector: Current global model estimate.

        Returns:
            Updated trust score.
        """
        if global_vector is None:
            self.logger.debug(
                f"Client {client_id}: No global vector yet, using initial trust"
            )
            self.trust_history[client_id].append(self.trust_scores[client_id])
            return self.trust_scores[client_id]

        if update_vector is None or len(update_vector) == 0:
            self.logger.warning(
                f"Client {client_id}: Empty update vector, applying penalty"
            )
            new_trust = self.trust_scores[client_id] * self.soft_decay
            new_trust = float(np.clip(new_trust, 0.0, 1.0))
            self.trust_scores[client_id] = new_trust
            self.trust_history[client_id].append(new_trust)
            return new_trust

        similarity = self._compute_similarity(update_vector, global_vector)

        trust_update = self.update_alpha * similarity

        current_trust = self.trust_scores[client_id]
        new_trust = current_trust + trust_update

        new_trust = float(np.clip(new_trust, 0.0, 1.0))

        if self.quarantined.get(client_id, False):
            if new_trust > self.quarantine_threshold:
                self.quarantined[client_id] = False
                self.logger.info(f"Client {client_id} removed from quarantine")

        if new_trust < self.quarantine_threshold:
            if not self.quarantined.get(client_id, False):
                self.quarantined[client_id] = True
                self.logger.warning(
                    f"Client {client_id} quarantined (trust={new_trust:.3f})"
                )

            new_trust *= self.soft_decay

        self.trust_scores[client_id] = new_trust
        self.trust_history[client_id].append(new_trust)

        return new_trust

    def _compute_similarity(
        self,
        update: np.ndarray,
        global_vec: np.ndarray,
    ) -> float:
        """Compute cosine similarity between update and global vector."""
        norm_update = np.linalg.norm(update)
        norm_global = np.linalg.norm(global_vec)

        if norm_update < 1e-8 or norm_global < 1e-8:
            return 0.5

        cos_sim = np.dot(update, global_vec) / (norm_update * norm_global)

        return (cos_sim + 1) / 2

    def get_trust(self, client_id: str) -> float:
        """Get current trust score for a client."""
        return self.trust_scores[client_id]

    def is_quarantined(self, client_id: str) -> bool:
        """Check if client is quarantined."""
        return self.quarantined.get(client_id, False)

    def get_stats(self) -> dict[str, Any]:
        """Get trust statistics."""
        trust_values = list(self.trust_scores.values())

        if not trust_values:
            return {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "quarantined_count": 0,
            }

        return {
            "mean": float(np.mean(trust_values)),
            "std": float(np.std(trust_values)),
            "min": float(np.min(trust_values)),
            "max": float(np.max(trust_values)),
            "quarantined_count": sum(self.quarantined.values()),
        }

    def get_all_trust_scores(self) -> dict[str, float]:
        """Get all trust scores."""
        return dict(self.trust_scores)

    def get_history(self, client_id: str) -> list[float]:
        """Get trust history for a client."""
        return self.trust_history.get(client_id, [])

    def reset(self) -> None:
        """Reset all trust scores."""
        self.trust_scores.clear()
        self.quarantined.clear()
        self.trust_history.clear()
