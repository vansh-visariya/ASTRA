"""
Model Zoo for Federated Learning.

Provides CNN models for MNIST/CIFAR and utilities for model creation.
"""

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """Simple CNN for MNIST."""
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class CIFAR10CNN(nn.Module):
    """CNN for CIFAR-10."""
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(64 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class SimpleMLP(nn.Module):
    """Simple MLP for basic experiments."""
    
    def __init__(self, input_dim: int = 784, num_classes: int = 10, hidden_dim: int = 256):
        super().__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


def create_model(config: Dict[str, Any]) -> nn.Module:
    """
    Create model based on configuration.
    
    Args:
        config: Model configuration.
    
    Returns:
        Instantiated model.
    """
    model_config = config.get('model', {})
    model_type = model_config.get('type', 'cnn')
    
    if model_type == 'cnn':
        cnn_config = model_config.get('cnn', {})
        cnn_name = cnn_config.get('name', 'simple_cnn')
        
        dataset = config.get('dataset', {}).get('name', 'MNIST')
        
        if dataset == 'CIFAR10':
            return CIFAR10CNN(num_classes=10)
        else:
            return SimpleCNN(num_classes=10)
    
    elif model_type == 'mlp':
        return SimpleMLP()
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_model_num_params(model: nn.Module) -> int:
    """Get total number of parameters in model."""
    return sum(p.numel() for p in model.parameters())


def get_model_size_mb(model: nn.Module) -> float:
    """Get model size in megabytes."""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    
    return (param_size + buffer_size) / (1024 ** 2)


def flatten_model_params(model: nn.Module) -> torch.Tensor:
    """Flatten all model parameters to a single tensor."""
    return torch.cat([p.data.flatten() for p in model.parameters()])


def load_model_weights(model: nn.Module, state_dict: Dict[str, Any]) -> None:
    """Load weights into model."""
    model.load_state_dict(state_dict)
