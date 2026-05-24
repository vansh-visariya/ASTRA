"""Aggregation strategies — FedAvg, Trimmed Mean, Coordinate Median, Hybrid."""

from astra.core.aggregation.aggregator import (
    Aggregator,
    FedAvgAggregator,
    RobustAggregator,
    create_aggregator,
)
from astra.core.aggregation.robust import coordinate_median, hybrid_aggregator, trimmed_mean

__all__ = [
    "Aggregator",
    "FedAvgAggregator",
    "RobustAggregator",
    "create_aggregator",
    "trimmed_mean",
    "coordinate_median",
    "hybrid_aggregator",
]
