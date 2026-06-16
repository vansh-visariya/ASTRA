"""
Shared test fixtures.
"""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from astra.core.config import DEFAULT_CONFIG


class TinyModel(torch.nn.Module):
    def __init__(self, input_dim=10, hidden_dim=5, num_classes=3):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


@pytest.fixture
def sample_config():
    return {
        "seed": 42,
        "dataset": {"name": "MNIST", "split": "iid"},
        "model": {},
        "client": {"num_clients": 5, "local_epochs": 1, "batch_size": 16, "lr": 0.01},
        "server": {
            "optimizer": "sgd",
            "server_lr": 0.5,
            "momentum": 0.9,
            "async_lambda": 0.2,
            "aggregator_window": 10,
            "adaptive_lr": False,
            "lr_decay_factor": 0.5,
            "instability_threshold": 0.15,
        },
        "robust": {"method": "fedavg", "trim_ratio": 0.1},
        "trust": {
            "init": 1.0,
            "update_alpha": 0.3,
            "quarantine_threshold": 0.35,
            "soft_decay": 0.8,
        },
        "malicious": {"enabled": False, "ratio": 0.0, "behaviors": []},
        "privacy": {
            "dp_enabled": False,
            "dp_mode": "client",
            "clip_norm": 1.0,
            "sigma": 1.2,
        },
        "communication": {"compression": "none"},
        "peft": {"enabled": False},
    }


def _make_tiny_config(**overrides):
    data = {
        "seed": 42,
        "dataset": {"name": "MNIST", "split": "iid"},
        "model": {},
        "client": {"num_clients": 5, "local_epochs": 1, "batch_size": 16, "lr": 0.01},
        "server": {
            "optimizer": "sgd",
            "server_lr": 0.5,
            "momentum": 0.9,
            "async_lambda": 0.2,
            "aggregator_window": 10,
            "adaptive_lr": False,
            "lr_decay_factor": 0.5,
            "instability_threshold": 0.15,
        },
        "robust": {"method": "fedavg", "trim_ratio": 0.1},
        "trust": {
            "init": 1.0,
            "update_alpha": 0.3,
            "quarantine_threshold": 0.35,
            "soft_decay": 0.8,
        },
        "malicious": {"enabled": False, "ratio": 0.0, "behaviors": []},
        "privacy": {
            "dp_enabled": False,
            "dp_mode": "client",
            "clip_norm": 1.0,
            "sigma": 1.2,
        },
        "communication": {"compression": "none"},
        "peft": {"enabled": False},
    }
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(data.get(k), dict):
            data[k].update(v)
        else:
            data[k] = v
    return data


@pytest.fixture
def dummy_data():
    x = torch.randn(100, 10)
    y = torch.randint(0, 3, (100,))
    return TensorDataset(x, y)


@pytest.fixture
def val_loader():
    x = torch.randn(20, 10)
    y = torch.randint(0, 3, (20,))
    return DataLoader(TensorDataset(x, y), batch_size=8)


@pytest.fixture
def tiny_model_factory():
    def _factory():
        return TinyModel()

    return _factory


@pytest.fixture(autouse=True)
def _reset_registry():
    from astra.infra.registry import _global_registry

    _global_registry = None
    yield
    _global_registry = None
