"""
Simulation utilities for federated learning.
"""

import logging
import random
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch


class NetworkSimulator:
    """Simulates network latency for client-server communication."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('simulation', {})
        
        self.latency_mean = self.config.get('latency_mean', 0.5)
        self.latency_std = self.config.get('latency_std', 0.2)
        
        self.enabled = self.latency_mean > 0
    
    def simulate_latency(self) -> float:
        """Simulate network latency in seconds."""
        if not self.enabled:
            return 0.0
        
        return max(0.01, np.random.normal(self.latency_mean, self.latency_std))


class ClientSimulator:
    """Simulates client behavior including stragglers."""
    
    def __init__(self, client_id: str, config: Dict[str, Any]):
        self.client_id = client_id
        self.config = config
        
        self.is_straggler = self._determine_straggler()
        self.straggler_delay = self._determine_delay() if self.is_straggler else 0.0
    
    def _determine_straggler(self) -> bool:
        """Determine if client is a straggler."""
        straggler_prob = self.config.get('simulation', {}).get('straggler_ratio', 0.2)
        return random.random() < straggler_prob
    
    def _determine_delay(self) -> float:
        """Determine straggler delay."""
        delay_config = self.config.get('simulation', {})
        return delay_config.get('straggler_delay', 5.0)


class SlowClientSimulator:
    """Simulates slow clients with configurable latency distribution."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        self.slow_ratio = config.get('simulation', {}).get('slow_ratio', 0.3)
        self.slow_latency = config.get('simulation', {}).get('slow_latency', 2.0)
    
    def get_latency(self, client_id: str) -> float:
        """Get simulated latency for client."""
        if random.random() < self.slow_ratio:
            return self.slow_latency
        
        return random.uniform(0.1, 0.5)


def simulate_client_update(
    client_id: str,
    model_factory: Callable,
    data_loader: Any,
    config: Dict[str, Any],
    network_simulator: Optional[NetworkSimulator] = None
) -> Dict[str, Any]:
    """
    Simulate client update with network latency.
    
    Args:
        client_id: Client identifier.
        model_factory: Function to create model.
        data_loader: Client data loader.
        config: Configuration.
        network_simulator: Network simulator.
    
    Returns:
        Client update dictionary.
    """
    if network_simulator and network_simulator.enabled:
        latency = network_simulator.simulate_latency()
        time.sleep(latency)
    
    from core_engine.client import FLClient
    
    client = FLClient(
        client_id=client_id,
        train_data=data_loader,
        model_factory=model_factory,
        config=config
    )
    
    update = client.local_train()
    
    return update


def simulate_round(
    clients: List[Any],
    aggregator: Any,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Simulate one round of federated learning.
    
    Args:
        clients: List of clients.
        aggregator: Aggregator instance.
        config: Configuration.
    
    Returns:
        Round results.
    """
    updates = []
    
    for client in clients:
        update = client.local_train()
        updates.append(update)
    
    aggregated = aggregator.aggregate(updates)
    
    return {
        'num_updates': len(updates),
        'aggregated': aggregated
    }
