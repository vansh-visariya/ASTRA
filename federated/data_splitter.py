"""
Data Splitting for Federated Learning.

Implements:
- Dirichlet distribution for non-IID data splitting
- Class imbalance simulation
- Per-client class distribution tracking

References:
- Hsu et al., "Measuring the Effects of Non-IID Data on Deep Learning"
- Yurochkin et al., "Bayesian Nonparametric Supervised Learning"
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms


class DataSplitter:
    """Handles data splitting for federated learning."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.dataset_config = config.get('dataset', {})
        
        self.dataset_name = self.dataset_config.get('name', 'MNIST')
        self.split_method = self.dataset_config.get('split', 'dirichlet')
        self.dirichlet_alpha = self.dataset_config.get('dirichlet_alpha', 0.3)
        self.imbalance = self.dataset_config.get('imbalance', True)
        
        self.num_clients = config.get('client', {}).get('num_clients', 20)
        
        self.logger = logging.getLogger(__name__)
        
        self.client_data: Dict[int, Subset] = {}
        self.class_distributions: Dict[int, Dict[int, float]] = {}
        
        self._load_dataset()
    
    def _load_dataset(self) -> None:
        """Load the base dataset."""
        if self.dataset_name == 'MNIST':
            self._load_mnist()
        elif self.dataset_name == 'CIFAR10':
            self._load_cifar10()
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
    
    def _load_mnist(self) -> None:
        """Load MNIST dataset."""
        data_dir = './data/mnist'
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        self.train_dataset = datasets.MNIST(
            data_dir, train=True, download=True, transform=transform
        )
        self.val_dataset = datasets.MNIST(
            data_dir, train=False, download=True, transform=transform
        )
    
    def _load_cifar10(self) -> None:
        """Load CIFAR-10 dataset."""
        data_dir = './data/cifar10'
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        self.train_dataset = datasets.CIFAR10(
            data_dir, train=True, download=True, transform=transform
        )
        self.val_dataset = datasets.CIFAR10(
            data_dir, train=False, download=True, transform=transform
        )
    
    def split_data(self) -> None:
        """Split data among clients using configured method."""
        if self.split_method == 'dirichlet':
            self._dirichlet_split()
        elif self.split_method == 'iid':
            self._iid_split()
        else:
            raise ValueError(f"Unknown split method: {self.split_method}")
    
    def _iid_split(self) -> None:
        """IID random split."""
        n_samples = len(self.train_dataset)
        indices = np.random.permutation(n_samples)
        
        samples_per_client = n_samples // self.num_clients
        
        for client_id in range(self.num_clients):
            start_idx = client_id * samples_per_client
            end_idx = start_idx + samples_per_client
            
            if client_id == self.num_clients - 1:
                end_idx = n_samples
            
            client_indices = indices[start_idx:end_idx]
            self.client_data[client_id] = Subset(self.train_dataset, client_indices)
    
    def _dirichlet_split(self) -> None:
        """Dirichlet non-IID split."""
        labels = self.train_dataset.targets
        num_classes = len(torch.unique(torch.tensor(list(labels))))
        
        client_distributions = np.random.dirichlet(
            [self.dirichlet_alpha] * self.num_clients,
            size=num_classes
        )
        
        if self.imbalance:
            imbalance_factors = np.random.uniform(0.5, 1.5, size=self.num_clients)
            client_distributions = client_distributions * imbalance_factors
            client_distributions = client_distributions / client_distributions.sum(axis=1, keepdims=True)
        
        self.client_data = {i: [] for i in range(self.num_clients)}
        
        class_indices = {c: [] for c in range(num_classes)}
        
        for idx, label in enumerate(labels):
            label_int = int(label) if hasattr(label, 'item') else label
            class_indices[label_int].append(idx)
        
        for class_id in range(num_classes):
            class_indices_list = class_indices[class_id]
            np.random.shuffle(class_indices_list)
            
            class_counts = (client_distributions[class_id] * len(class_indices_list)).astype(int)
            
            remaining = len(class_indices_list) - class_counts.sum()
            class_counts[np.argmax(class_counts)] += remaining
            
            start = 0
            for client_id in range(self.num_clients):
                end = start + class_counts[client_id]
                self.client_data[client_id].extend(class_indices_list[start:end])
                start = end
        
        for client_id in range(self.num_clients):
            np.random.shuffle(self.client_data[client_id])
            self.client_data[client_id] = Subset(
                self.train_dataset,
                self.client_data[client_id]
            )
        
        self._compute_class_distributions()
    
    def _compute_class_distributions(self) -> None:
        """Compute class distribution for each client."""
        for client_id, subset in self.client_data.items():
            indices = subset.indices
            labels = self.train_dataset.targets[indices]
            
            unique, counts = torch.unique(torch.tensor(list(labels)), return_counts=True)
            
            total = len(labels)
            distribution = {int(u): int(c) / total for u, c in zip(unique, counts)}
            
            self.class_distributions[client_id] = distribution
    
    def get_client_data(self, client_id: int) -> Subset:
        """Get data for a specific client."""
        if not self.client_data:
            self.split_data()
        return self.client_data[client_id]
    
    def create_data_loaders(self) -> Tuple[Dict[int, DataLoader], DataLoader]:
        """Create data loaders for all clients and validation set."""
        if not self.client_data:
            self.split_data()
        
        batch_size = self.config.get('client', {}).get('batch_size', 32)
        
        client_loaders = {}
        for client_id, subset in self.client_data.items():
            client_loaders[client_id] = DataLoader(
                subset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0
            )
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        
        return client_loaders, val_loader
    
    def get_class_distribution(self, client_id: int) -> Dict[int, float]:
        """Get class distribution for a client."""
        return self.class_distributions.get(client_id, {})
    
    def save_distributions(self, path: Path) -> None:
        """Save class distributions to file."""
        with open(path, 'w') as f:
            json.dump(self.class_distributions, f, indent=2)
