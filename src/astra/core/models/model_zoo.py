"""
Model Zoo for Federated Learning.

Provides CNN models for MNIST/CIFAR and utilities for model creation.
"""

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812


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


def create_model(config: dict[str, Any]) -> nn.Module:
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
        cnn_config.get('name', 'simple_cnn')

        dataset_cfg = config.get('dataset', {})
        if isinstance(dataset_cfg, str):
            dataset_cfg = {'name': dataset_cfg}
        dataset = dataset_cfg.get('name', 'MNIST')

        if dataset == 'CIFAR10':
            return CIFAR10CNN(num_classes=10)
        else:
            return SimpleCNN(num_classes=10)

    elif model_type == 'mlp':
        return SimpleMLP()

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def _is_lora_param(name: str) -> bool:
    """Check if a parameter name belongs to a LoRA/adapter module."""
    return "lora" in name.lower() or "adapter" in name.lower()


def flatten_peft_params(model: nn.Module) -> np.ndarray:
    """Flatten only LoRA/adapter parameters to a flat numpy array.

    Parameters are processed in sorted name order for deterministic ordering
    across all clients and the server.
    """
    import numpy as np

    peft_params = sorted(
        [(name, param) for name, param in model.named_parameters() if _is_lora_param(name)],
        key=lambda x: x[0],
    )
    if not peft_params:
        return np.array([], dtype=np.float32)
    return np.concatenate(
        [param.data.cpu().numpy().flatten().astype(np.float32) for _, param in peft_params]
    )


def apply_peft_delta(model: nn.Module, flat_delta: np.ndarray) -> None:
    """Apply a flat LoRA delta to the model's LoRA parameters in-place."""
    offset = 0
    for _name, param in sorted(
        [(n, p) for n, p in model.named_parameters() if _is_lora_param(n)],
        key=lambda x: x[0],
    ):
        size = param.numel()
        if offset + size <= len(flat_delta):
            delta_slice = flat_delta[offset : offset + size].reshape(param.shape)
            param.data.add_(torch.from_numpy(delta_slice).float().to(param.device))
        offset += size
