"""
Unit tests for metrics module: compute_accuracy, compute_loss, gradient_norm,
weight_norm, weight_stats, similarity, trust_metrics, attack_metrics, MetricsTracker.
"""

import numpy as np
import torch
import torch.nn as nn
import pytest
from torch.utils.data import DataLoader, TensorDataset

from astra.core.utils.metrics import (
    compute_accuracy,
    compute_loss,
    compute_gradient_norm,
    compute_weight_norm,
    compute_weight_stats,
    compute_similarity,
    compute_trust_metrics,
    compute_attack_metrics,
    MetricsTracker,
)


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 3)

    def forward(self, x):
        return self.fc(x)


class TestComputeAccuracy:
    def test_perfect_accuracy(self):
        model = DummyModel()
        with torch.no_grad():
            model.fc.weight.fill_(0)
            model.fc.bias[0] = 99.0

        x = torch.randn(20, 10)
        y = torch.zeros(20, dtype=torch.long)
        loader = DataLoader(TensorDataset(x, y), batch_size=8)

        acc = compute_accuracy(model, loader)
        assert acc == 1.0

    def test_zero_accuracy(self):
        model = DummyModel()
        with torch.no_grad():
            model.fc.weight.fill_(0)
            model.fc.bias[0] = 99.0
            model.fc.bias[1] = 0.0
            model.fc.bias[2] = 0.0

        x = torch.randn(20, 10)
        y = torch.ones(20, dtype=torch.long)
        loader = DataLoader(TensorDataset(x, y), batch_size=8)

        acc = compute_accuracy(model, loader)
        assert acc == 0.0

    def test_partial_accuracy(self):
        model = DummyModel()
        x = torch.randn(20, 10)
        y = torch.randint(0, 3, (20,))
        loader = DataLoader(TensorDataset(x, y), batch_size=8)

        acc = compute_accuracy(model, loader)
        assert 0.0 <= acc <= 1.0

    def test_empty_loader(self):
        model = DummyModel()
        x = torch.randn(0, 10)
        y = torch.zeros(0, dtype=torch.long)
        loader = DataLoader(TensorDataset(x, y))

        acc = compute_accuracy(model, loader)
        assert acc == 0.0


class TestComputeLoss:
    def test_loss_is_positive(self):
        model = DummyModel()
        x = torch.randn(20, 10)
        y = torch.randint(0, 3, (20,))
        loader = DataLoader(TensorDataset(x, y), batch_size=8)

        loss = compute_loss(model, loader)
        assert loss > 0.0

    def test_empty_loader_loss(self):
        model = DummyModel()
        x = torch.randn(0, 10)
        y = torch.zeros(0, dtype=torch.long)
        loader = DataLoader(TensorDataset(x, y))

        loss = compute_loss(model, loader)
        assert loss == 0.0

    def test_multi_batch_loss(self):
        model = DummyModel()
        x = torch.randn(32, 10)
        y = torch.randint(0, 3, (32,))
        loader = DataLoader(TensorDataset(x, y), batch_size=8)

        loss = compute_loss(model, loader)
        assert loss > 0.0


class TestGradientNorm:
    def test_no_gradients_returns_zero(self):
        model = DummyModel()
        norm = compute_gradient_norm(model)
        assert norm == 0.0

    def test_gradient_norm_after_backward(self):
        model = DummyModel()
        x = torch.randn(4, 10)
        y = torch.randint(0, 3, (4,))

        out = model(x)
        loss = nn.CrossEntropyLoss()(out, y)
        loss.backward()

        norm = compute_gradient_norm(model)
        assert norm > 0.0


class TestWeightNorm:
    def test_weight_norm_positive(self):
        model = DummyModel()
        norm = compute_weight_norm(model)
        assert norm > 0.0

    def test_zero_weights(self):
        model = DummyModel()
        with torch.no_grad():
            for p in model.parameters():
                p.zero_()
        norm = compute_weight_norm(model)
        assert norm == 0.0


class TestWeightStats:
    def test_stats_keys(self):
        w = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        stats = compute_weight_stats(w)
        assert set(stats.keys()) == {"mean", "std", "min", "max", "norm", "abs_mean"}
        assert stats["mean"] == 3.0
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0

    def test_identical_values(self):
        w = np.array([2.0, 2.0, 2.0])
        stats = compute_weight_stats(w)
        assert stats["std"] == 0.0
        assert stats["mean"] == 2.0

    def test_single_element(self):
        w = np.array([7.0])
        stats = compute_weight_stats(w)
        assert stats["mean"] == 7.0
        assert stats["std"] == 0.0


class TestComputeSimilarity:
    def test_identical_vectors(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 3.0])
        sim = compute_similarity(a, b)
        assert sim["cosine"] == pytest.approx(1.0, abs=1e-5)
        assert sim["euclidean"] == pytest.approx(0.0, abs=1e-5)
        assert sim["mse"] == pytest.approx(0.0, abs=1e-5)

    def test_opposite_vectors(self):
        a = np.array([1.0, 1.0, 1.0])
        b = np.array([-1.0, -1.0, -1.0])
        sim = compute_similarity(a, b)
        assert sim["cosine"] == pytest.approx(-1.0, abs=1e-5)

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        sim = compute_similarity(a, b)
        assert sim["cosine"] == pytest.approx(0.0, abs=1e-5)

    def test_zero_vector(self):
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([1.0, 2.0, 3.0])
        sim = compute_similarity(a, b)
        assert sim["cosine"] == 0.0


class TestTrustMetrics:
    def test_empty_scores(self):
        result = compute_trust_metrics({})
        assert result["mean"] == 0.0
        assert result["min"] == 0.0

    def test_single_score(self):
        result = compute_trust_metrics({"c1": 0.8})
        assert result["mean"] == 0.8
        assert result["min"] == 0.8
        assert result["max"] == 0.8

    def test_multiple_scores(self):
        result = compute_trust_metrics({"a": 0.5, "b": 0.8, "c": 0.3})
        assert result["min"] == 0.3
        assert result["max"] == 0.8


class TestAttackMetrics:
    def test_large_drop(self):
        result = compute_attack_metrics(0.9, 0.4, 50)
        assert result["attack_success"] == 1.0
        assert result["accuracy_drop"] == 0.5

    def test_small_drop(self):
        result = compute_attack_metrics(0.9, 0.88, 50)
        assert result["attack_success"] == 0.0

    def test_no_drop(self):
        result = compute_attack_metrics(0.9, 0.9, 10)
        assert result["accuracy_drop"] == 0.0
        assert result["relative_drop"] == 0.0

    def test_zero_pre_attack(self):
        result = compute_attack_metrics(0.0, 0.0, 5)
        assert result["relative_drop"] == 0.0
