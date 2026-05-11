"""
Aggregator Interface and Implementations.

Provides the base Aggregator class and factory method for creating
different aggregation strategies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from core_engine.exceptions import AggregationError, ConfigurationError
from core_engine.robust_aggregation import (
    coordinate_median,
    hybrid_aggregator,
    trimmed_mean,
)

__all__ = [
    "Aggregator",
    "FedAvgAggregator",
    "RobustAggregator",
    "create_aggregator",
]


class Aggregator(ABC):
    """Base class for federated learning aggregators."""

    @abstractmethod
    def aggregate(self, buffer: list[dict[str, Any]]) -> np.ndarray:
        """
        Aggregate client updates.

        Args:
            buffer: List of client updates, each containing:
                - client_id: str
                - delta: np.ndarray of weight changes
                - staleness_weight: float
                - trust: float
                - local_dataset_size: int

        Returns:
            Aggregated weight delta as numpy array.

        Raises:
            AggregationError: If aggregation fails.
        """
        pass


class FedAvgAggregator(Aggregator):
    """Standard Federated Averaging (FedAvg) aggregator."""

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize FedAvg aggregator.

        Args:
            config: Configuration dictionary.

        Raises:
            ConfigurationError: If configuration is invalid.
        """
        if not isinstance(config, dict):
            raise ConfigurationError("Config must be a dictionary")
        self.config = config

    def aggregate(self, buffer: list[dict[str, Any]]) -> np.ndarray:
        """
        Perform weighted FedAvg aggregation.

        Args:
            buffer: List of client updates.

        Returns:
            Aggregated weight delta.

        Raises:
            AggregationError: If buffer is empty or aggregation fails.
        """
        if not buffer:
            raise AggregationError("Cannot aggregate empty buffer")

        total_weight: float = 0.0
        aggregated: np.ndarray | None = None

        for update in buffer:
            delta = np.nan_to_num(
                update["delta"], nan=0.0, posinf=1e6, neginf=-1e6
            )
            dataset_size = update.get("local_dataset_size", 1)
            weight = dataset_size * update.get("staleness_weight", 1.0)

            if aggregated is None:
                aggregated = weight * delta
            else:
                aggregated = aggregated + weight * delta

            total_weight += weight

        if total_weight > 0 and aggregated is not None:
            aggregated = aggregated / total_weight

        if aggregated is not None:
            aggregated = np.nan_to_num(
                aggregated, nan=0.0, posinf=1e6, neginf=-1e6
            )

        return aggregated if aggregated is not None else np.array([])


class RobustAggregator(Aggregator):
    """Robust aggregator using Byzantine-tolerant methods."""

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize robust aggregator.

        Args:
            config: Configuration dictionary.

        Raises:
            ConfigurationError: If configuration is invalid.
        """
        if not isinstance(config, dict):
            raise ConfigurationError("Config must be a dictionary")

        robust_config = config.get("robust", {})
        if not isinstance(robust_config, dict):
            raise ConfigurationError("robust config must be a dictionary")

        self.config = config
        self.method = robust_config.get("method", "median")

    def aggregate(self, buffer: list[dict[str, Any]]) -> np.ndarray:
        """
        Aggregate using robust method.

        Args:
            buffer: List of client updates.

        Returns:
            Aggregated weight delta.

        Raises:
            AggregationError: If buffer is empty or method is unknown.
        """
        if not buffer:
            raise AggregationError("Cannot aggregate empty buffer")

        deltas = [update["delta"] for update in buffer]
        trust_scores = [update.get("trust", 1.0) for update in buffer]
        staleness_weights = [update.get("staleness_weight", 1.0) for update in buffer]
        dataset_sizes = [update.get("local_dataset_size", 1) for update in buffer]

        if self.method == "trimmed_mean":
            trim_ratio = self.config.get("robust", {}).get("trim_ratio", 0.1)
            return trimmed_mean(deltas, trim_ratio)

        if self.method == "median":
            return coordinate_median(deltas)

        if self.method == "hybrid":
            return hybrid_aggregator(
                deltas,
                trust_scores,
                staleness_weights,
                self.config,
                dataset_sizes,
            )

        raise AggregationError(f"Unknown robust method: {self.method}")


def create_aggregator(config: dict[str, Any]) -> Aggregator:
    """
    Factory function to create appropriate aggregator.

    Args:
        config: Configuration dictionary.

    Returns:
        Aggregator instance.

    Raises:
        ConfigurationError: If configuration is invalid.
    """
    if not isinstance(config, dict):
        raise ConfigurationError("Config must be a dictionary")

    robust_method = config.get("robust", {}).get("method")

    if robust_method is not None and robust_method != "fedavg":
        return RobustAggregator(config)
    return FedAvgAggregator(config)
