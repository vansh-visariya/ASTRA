"""
Unit tests for model_zoo: flatten/apply functions, create_model, registry integration.
"""

import numpy as np
import torch
import torch.nn as nn
import pytest

from astra.core.models.model_zoo import (
    SimpleCNN,
    CIFAR10CNN,
    SimpleMLP,
    create_model,
    flatten_all_params,
    apply_flat_delta,
    flatten_peft_params,
    apply_peft_delta,
    _is_lora_param,
)


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
        self.b = nn.Parameter(torch.tensor([4.0, 5.0]))
        self.lora_A = nn.Parameter(torch.tensor([0.1, 0.2]))
        self.adapter_down = nn.Parameter(torch.tensor([0.3]))

    def forward(self, x):
        return x


class TestFlattenAllParams:
    def test_simple_cnn_roundtrip(self):
        model = SimpleCNN(num_classes=10)
        flat = flatten_all_params(model)
        assert flat.dtype == np.float32
        assert len(flat) > 0

    def test_cifar10_cnn_roundtrip(self):
        model = CIFAR10CNN(num_classes=10)
        flat = flatten_all_params(model)
        assert len(flat) > 0

    def test_flatten_then_apply_deterministic(self):
        model = DummyModel()
        original = flatten_all_params(model)

        delta = np.ones_like(original)
        apply_flat_delta(model, delta)

        updated = flatten_all_params(model)
        assert np.allclose(updated, original + 1.0)

    def test_delta_too_short_handled_gracefully(self):
        model = DummyModel()
        original = flatten_all_params(model)
        apply_flat_delta(model, np.array([99.0]))
        assert np.allclose(flatten_all_params(model), original + 0.0)

    def test_zero_delta_no_change(self):
        model = DummyModel()
        original = flatten_all_params(model)
        apply_flat_delta(model, np.zeros_like(original))
        assert np.allclose(flatten_all_params(model), original)

    def test_small_delta(self):
        model = DummyModel()
        original = flatten_all_params(model)
        apply_flat_delta(model, np.full_like(original, 0.001))
        assert np.allclose(flatten_all_params(model), original + 0.001)

    def test_model_unchanged_weights_unchanged(self):
        torch.manual_seed(42)
        model1 = SimpleCNN(num_classes=10)
        torch.manual_seed(42)
        model2 = SimpleCNN(num_classes=10)
        flat1 = flatten_all_params(model1)
        flat2 = flatten_all_params(model2)
        assert np.allclose(flat1, flat2)


class TestPEFTFlatten:
    def test_is_lora_param_detection(self):
        assert _is_lora_param("model.lora_A.weight") is True
        assert _is_lora_param("adapter_down.bias") is True
        assert _is_lora_param("ADAPTER_UP") is True
        assert _is_lora_param("conv1.weight") is False
        assert _is_lora_param("fc.bias") is False

    def test_flatten_peft_params_only_lora(self):
        model = DummyModel()
        flat = flatten_peft_params(model)
        assert len(flat) == 3

    def test_apply_peft_delta_changes_only_lora(self):
        model = DummyModel()
        a_before = model.a.data.clone()
        b_before = model.b.data.clone()

        peft_flat = flatten_peft_params(model)
        apply_peft_delta(model, np.ones_like(peft_flat) * 2.0)

        assert torch.equal(model.a.data, a_before)
        assert torch.equal(model.b.data, b_before)
        assert not torch.equal(model.lora_A.data, torch.tensor([0.1, 0.2]))

    def test_no_peft_params_handled(self):
        model = nn.Linear(10, 10)
        flat = flatten_peft_params(model)
        assert len(flat) == 0
        apply_peft_delta(model, np.array([1.0]))


class TestCreateModel:
    def test_default_creates_simple_cnn(self):
        from astra.infra.registry import get_registry
        model = get_registry().build_model("simple_cnn_mnist")
        assert isinstance(model, SimpleCNN)
        assert model.fc2.out_features == 10

    def test_cifar10_dataset_creates_cifar10_cnn(self):
        from astra.infra.registry import get_registry
        model = get_registry().build_model("simple_cnn_cifar10")
        assert isinstance(model, CIFAR10CNN)

    def test_legacy_fallback_creates_simple_cnn(self):
        config = {"model": {"type": "cnn", "model_id": "NONEXISTENT_xyz"}, "dataset": {"name": "MNIST"}}
        model = create_model(config)
        assert isinstance(model, SimpleCNN)

    def test_legacy_fallback_creates_cifar10(self):
        config = {"model": {"type": "cnn", "model_id": "NONEXISTENT_xyz"}, "dataset": {"name": "CIFAR10"}}
        model = create_model(config)
        assert isinstance(model, CIFAR10CNN)

    def test_legacy_fallback_creates_mlp(self):
        config = {"model": {"type": "mlp", "model_id": "NONEXISTENT_xyz"}, "dataset": {"name": "MNIST"}}
        model = create_model(config)
        assert isinstance(model, SimpleMLP)

    def test_forward_pass_works(self):
        model = SimpleCNN(num_classes=10)
        x = torch.randn(2, 1, 28, 28)
        out = model(x)
        assert out.shape == (2, 10)

    def test_cifar10_forward_pass(self):
        model = CIFAR10CNN(num_classes=10)
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        assert out.shape == (2, 10)

    def test_mlp_forward_pass(self):
        model = SimpleMLP(input_dim=784, num_classes=10)
        x = torch.randn(2, 784)
        out = model(x)
        assert out.shape == (2, 10)
