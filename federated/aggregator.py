"""
Aggregator Interface and Implementations.

Provides the base Aggregator class and factory method for creating
different aggregation strategies.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

import numpy as np

from federated.robust_aggregation import (
    trimmed_mean,
    coordinate_median,
    hybrid_aggregator
)


class Aggregator(ABC):
    """Base class for federated learning aggregators."""
    
    @abstractmethod
    def aggregate(self, buffer: List[Dict[str, Any]]) -> np.ndarray:
        """
        Aggregate client updates.
        
        Args:
            buffer: List of client updates, each containing:
                - client_id: str
                - delta: np.ndarray of weight changes
                - staleness_weight: float
                - trust: float
        
        Returns:
            Aggregated weight delta as numpy array.
        """
        pass


class FedAvgAggregator(Aggregator):
    """Standard Federated Averaging (FedAvg) aggregator."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def aggregate(self, buffer: List[Dict[str, Any]]) -> np.ndarray:
        """Simple weighted average by dataset size."""
        if not buffer:
            return np.array([])
        
        total_weight = 0.0
        aggregated = None
        
        for update in buffer:
            delta = np.nan_to_num(update['delta'], nan=0.0, posinf=1e6, neginf=-1e6)
            dataset_size = update.get('dataset_size', 1)
            weight = dataset_size * update.get('staleness_weight', 1.0)
            
            if aggregated is None:
                aggregated = weight * delta
            else:
                aggregated += weight * delta
            
            total_weight += weight
        
        if total_weight > 0:
            aggregated /= total_weight
        
        aggregated = np.nan_to_num(aggregated, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return aggregated


class RobustAggregator(Aggregator):
    """Robust aggregator using specified method."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.method = config['robust'].get('method', 'median')
    
    def aggregate(self, buffer: List[Dict[str, Any]]) -> np.ndarray:
        """Aggregate using robust method."""
        if not buffer:
            return np.array([])
        
        deltas = [update['delta'] for update in buffer]
        trust_scores = [update.get('trust', 1.0) for update in buffer]
        staleness_weights = [update.get('staleness_weight', 1.0) for update in buffer]
        
        if self.method == 'trimmed_mean':
            trim_ratio = self.config['robust'].get('trim_ratio', 0.1)
            return trimmed_mean(deltas, trim_ratio)
        
        elif self.method == 'median':
            return coordinate_median(deltas)
        
        elif self.method == 'hybrid':
            return hybrid_aggregator(
                deltas,
                trust_scores,
                staleness_weights,
                self.config
            )
        
        else:
            raise ValueError(f"Unknown robust method: {self.method}")


def create_aggregator(config: Dict[str, Any]) -> Aggregator:
    """
    Factory function to create appropriate aggregator.
    
    Args:
        config: Configuration dictionary.
    
    Returns:
        Aggregator instance.
    """
    robust_enabled = config.get('robust', {}).get('method') is not None
    
    if robust_enabled and config['robust']['method'] != 'fedavg':
        return RobustAggregator(config)
    else:
        return FedAvgAggregator(config)
