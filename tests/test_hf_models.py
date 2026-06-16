"""
Unit tests for hf_models: freeze_backbone, LoRA state dict, base model state dict,
load_lora_state_dict.
"""

import torch
import torch.nn as nn
import pytest

from astra.core.models.hf_models import (
    freeze_backbone,
    get_lora_state_dict,
    load_lora_state_dict,
    get_base_model_state_dict,
)


class DummyLoRAModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3)
        self.lora_A = nn.Linear(16, 8)
        self.lora_B = nn.Linear(8, 16)
        self.adapter_down = nn.Linear(16, 4)
        self.adapter_up = nn.Linear(4, 16)
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        x = self.conv(x)
        x = x.mean(dim=[2, 3])
        x = self.adapter_up(self.adapter_down(x))
        x = self.lora_B(self.lora_A(x))
        return self.fc(x)


class TestFreezeBackbone:
    def test_only_lora_trainable(self):
        model = DummyLoRAModel()
        freeze_backbone(model)

        assert model.conv.weight.requires_grad is False
        assert model.fc.weight.requires_grad is False
        assert model.lora_A.weight.requires_grad is True
        assert model.lora_B.weight.requires_grad is True
        assert model.adapter_down.weight.requires_grad is True
        assert model.adapter_up.weight.requires_grad is True

    def test_no_lora_all_frozen(self):
        model = nn.Linear(10, 5)
        freeze_backbone(model)
        assert model.weight.requires_grad is False

    def test_all_lora_all_trainable(self):
        class AllLoRA(nn.Module):
            def __init__(self):
                super().__init__()
                self.lora_a = nn.Linear(10, 5)
                self.adapter = nn.Linear(5, 3)
            def forward(self, x):
                return self.adapter(self.lora_a(x))

        model = AllLoRA()
        freeze_backbone(model)
        assert model.lora_a.weight.requires_grad is True
        assert model.adapter.weight.requires_grad is True


class TestLoRAStateDict:
    def test_get_lora_only_returns_lora(self):
        model = DummyLoRAModel()
        state = get_lora_state_dict(model)
        assert all("lora" in k.lower() or "adapter" in k.lower() for k in state)
        assert len(state) == 8

    def test_get_base_model_excludes_lora(self):
        model = DummyLoRAModel()
        state = get_base_model_state_dict(model)
        assert not any("lora" in k.lower() or "adapter" in k.lower() for k in state)
        assert len(state) == 4

    def test_union_is_all_params(self):
        model = DummyLoRAModel()
        lora = get_lora_state_dict(model)
        base = get_base_model_state_dict(model)
        all_params = set(n for n, _ in model.named_parameters())
        union = set(lora.keys()) | set(base.keys())
        assert all_params == union

    def test_disjoint(self):
        model = DummyLoRAModel()
        lora = get_lora_state_dict(model)
        base = get_base_model_state_dict(model)
        assert set(lora.keys()).isdisjoint(set(base.keys()))


class TestLoadLoRAStateDict:
    def test_load_applies_correctly(self):
        model = DummyLoRAModel()
        lora_state = get_lora_state_dict(model)

        for k in lora_state:
            lora_state[k] = lora_state[k] + 1.0

        load_lora_state_dict(model, lora_state)

        for name, param in model.named_parameters():
            if name in lora_state:
                assert torch.allclose(param.data, lora_state[name].to(param.device))

    def test_load_empty_no_change(self):
        model = DummyLoRAModel()
        before = {n: p.data.clone() for n, p in model.named_parameters()}
        load_lora_state_dict(model, {})
        for n, p in model.named_parameters():
            assert torch.equal(p.data, before[n])
