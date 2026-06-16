"""
Unit tests for model_zoo: flatten/apply functions and SimpleMLP.
"""

import numpy as np
import torch
import torch.nn as nn

from astra.core.models.model_zoo import (
    SimpleMLP,
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


class TinyNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 3)

    def forward(self, x):
        return self.fc2(self.fc1(x))


class TestFlattenAllParams:
    def test_roundtrip_deterministic(self):
        torch.manual_seed(42)
        m1 = TinyNN()
        torch.manual_seed(42)
        m2 = TinyNN()
        assert np.allclose(flatten_all_params(m1), flatten_all_params(m2))

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

    def test_output_dtype(self):
        model = DummyModel()
        flat = flatten_all_params(model)
        assert flat.dtype == np.float32
        assert len(flat) > 0


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


class TestSimpleMLP:
    def test_forward_pass(self):
        model = SimpleMLP(input_dim=784, num_classes=10)
        x = torch.randn(2, 784)
        out = model(x)
        assert out.shape == (2, 10)

    def test_creates_correct_layers(self):
        model = SimpleMLP(input_dim=100, num_classes=5, hidden_dim=128)
        assert model.fc1.in_features == 100
        assert model.fc1.out_features == 128
        assert model.fc2.in_features == 128
        assert model.fc2.out_features == 128
        assert model.fc3.in_features == 128
        assert model.fc3.out_features == 5
