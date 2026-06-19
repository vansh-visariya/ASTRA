"""
Unit tests for metrics module: compute_accuracy, compute_loss.
"""

import torch
import torch.nn as nn
import pytest
from torch.utils.data import DataLoader, TensorDataset

from astra.core.utils.metrics import compute_accuracy, compute_loss


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 3)

    def forward(self, x):
        return self.fc(x)


@pytest.fixture
def dummy_loader():
    x = torch.randn(20, 10)
    y = torch.randint(0, 3, (20,))
    return DataLoader(TensorDataset(x, y), batch_size=5)


def test_compute_accuracy_returns_float():
    model = DummyModel()
    loader = DataLoader(TensorDataset(torch.randn(10, 10), torch.randint(0, 3, (10,))), batch_size=5)
    acc = compute_accuracy(model, loader)
    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0


def test_compute_loss_returns_positive():
    model = DummyModel()
    loader = DataLoader(TensorDataset(torch.randn(10, 10), torch.randint(0, 3, (10,))), batch_size=5)
    loss = compute_loss(model, loader)
    assert isinstance(loss, float)
    assert loss >= 0.0


def test_compute_accuracy_empty_loader():
    model = DummyModel()
    x = torch.randn(0, 10)
    y = torch.randint(0, 3, (0,))
    loader = DataLoader(TensorDataset(x, y), batch_size=5)
    acc = compute_accuracy(model, loader)
    assert acc == 0.0


def test_compute_loss_empty_loader():
    model = DummyModel()
    x = torch.randn(0, 10)
    y = torch.randint(0, 3, (0,))
    loader = DataLoader(TensorDataset(x, y), batch_size=5)
    loss = compute_loss(model, loader)
    assert loss == 0.0
