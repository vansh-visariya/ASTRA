"""
Federated Learning Client.

Implements local training on client devices and update generation.

References:
- McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data"
"""

import json
import pickle
import time
import logging
from typing import Any, Callable, Dict, Optional
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from core_engine.privacy import clip_and_noise, secure_aggregate_masking
from core_engine.malicious_simulator import MaliciousSimulator
from core_engine.compression import topk_sparsify


class FLClient:
    """Federated learning client."""
    
    def __init__(
        self,
        client_id: str,
        train_data: Any,
        model_factory: Callable[[], nn.Module],
        config: Dict[str, Any]
    ):
        self.client_id = client_id
        self.train_data = train_data
        self.model_factory = model_factory
        self.config = config
        
        self.model = model_factory()
        
        # Use GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        self.client_version = 0
        
        self.malicious_simulator = MaliciousSimulator(config)
        self.is_malicious = self._check_if_malicious()
        
        self.logger = logging.getLogger(__name__)
        
        self._init_optimizer()
        self._init_data_loader()
    
    def _check_if_malicious(self) -> bool:
        """Determine if this client is malicious based on config."""
        malicious_ratio = self.config['malicious'].get('ratio', 0)
        if malicious_ratio == 0:
            return False
        
        client_hash = hash(self.client_id)
        threshold = int(1 / malicious_ratio)
        return client_hash % threshold == 0
    
    def _init_optimizer(self):
        """Initialize client optimizer."""
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.config['client']['lr'],
            weight_decay=self.config['client'].get('weight_decay', 0.0)
        )
    
    def _init_data_loader(self):
        """Initialize data loader."""
        batch_size = self.config['client']['batch_size']
        
        if isinstance(self.train_data, Subset):
            self.train_loader = DataLoader(
                self.train_data,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0
            )
        else:
            self.train_loader = DataLoader(
                self.train_data,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0
            )
    
    def local_train(self) -> Dict[str, Any]:
        """
        Run local training on client data.
        
        Returns:
            Dictionary containing client update for server.
        """
        self.model.train()
        
        local_epochs = self.config['client']['local_epochs']
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        initial_weights = self._get_weights()
        
        for epoch in range(local_epochs):
            for batch_idx, (data, target) in enumerate(self.train_loader):
                # Move to GPU
                data, target = data.to(self.device), target.to(self.device)
                
                self.optimizer.zero_grad()
                
                output = self.model(data)
                loss = nn.functional.cross_entropy(output, target)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item() * len(target)
                pred = output.argmax(dim=1)
                total_correct += (pred == target).sum().item()
                total_samples += len(target)
        
        final_weights = self._get_weights()
        weight_delta = self._compute_weight_delta(initial_weights, final_weights)
        
        if self.is_malicious:
            weight_delta = self.malicious_simulator.inject_attack(weight_delta, self.client_id)
        
        if self.config['privacy']['dp_enabled'] and self.config['privacy']['dp_mode'] == 'client':
            weight_delta = clip_and_noise(
                weight_delta,
                self.config['privacy']['clip_norm'],
                self.config['privacy']['sigma']
            )
        
        if self.config['communication']['compression'] == 'topk':
            k_ratio = self.config['communication'].get('topk_ratio', 0.1)
            weight_delta, _ = topk_sparsify(weight_delta, k_ratio)
        
        self.client_version += 1
        
        train_loss = total_loss / total_samples if total_samples > 0 else 0.0
        train_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        
        update = {
            'client_id': self.client_id,
            'client_version': self.client_version,
            'local_updates': weight_delta.tobytes(),
            'update_type': 'delta',
            'local_dataset_size': len(self.train_data),
            'timestamp': time.time(),
            'meta': {
                'train_loss': train_loss,
                'train_accuracy': train_accuracy,
                'local_steps': local_epochs * len(self.train_loader)
            }
        }
        
        return update
    
    def _get_weights(self) -> np.ndarray:
        """Get model weights as flattened numpy array."""
        weights = []
        for param in self.model.parameters():
            weights.append(param.data.cpu().numpy().flatten())
        return np.concatenate(weights)
    
    def _compute_weight_delta(self, initial: np.ndarray, final: np.ndarray) -> np.ndarray:
        """Compute weight delta (final - initial)."""
        return final - initial
    
    def send_update(self, server: Any) -> None:
        """Send update to server (simulated networking)."""
        update = self.local_train()
        server.handle_update(update)
    
    def reset_model(self) -> None:
        """Reset model to initial state."""
        self.model = self.model_factory()
        self._init_optimizer()
    
    def get_client_stats(self) -> Dict[str, Any]:
        """Get client statistics."""
        return {
            'client_id': self.client_id,
            'is_malicious': self.is_malicious,
            'dataset_size': len(self.train_data),
            'client_version': self.client_version
        }
