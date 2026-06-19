"""
Unit tests for hf_models: freeze_backbone, LoRA state dict, base model state dict.
"""

import torch
import torch.nn as nn
import pytest

from astra.core.models.hf_models import (
    freeze_backbone,
    get_lora_state_dict,
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
        return self.fc(x)


@pytest.fixture
def model():
    return DummyLoRAModel()


class TestFreezeBackbone:
    def test_freeze_backbone(self, model):
        freeze_backbone(model)
        for name, param in model.named_parameters():
            if "lora" in name.lower() or "adapter" in name.lower():
                assert param.requires_grad is True
            else:
                assert param.requires_grad is False

    def test_freeze_backbone_all_params_exist(self, model):
        total = sum(p.numel() for p in model.parameters())
        assert total > 0


class TestGetLoraStateDict:
    def test_includes_lora(self, model):
        lora_state = get_lora_state_dict(model)
        assert len(lora_state) > 0
        for key in lora_state:
            assert "lora" in key.lower() or "adapter" in key.lower()

    def test_excludes_backbone(self, model):
        lora_state = get_lora_state_dict(model)
        for key in lora_state:
            assert "conv" not in key
            assert "fc" not in key


class TestGetBaseModelStateDict:
    def test_includes_backbone(self, model):
        base_state = get_base_model_state_dict(model)
        assert len(base_state) > 0
        assert any("conv" in key or "fc" in key for key in base_state)

    def test_excludes_lora(self, model):
        base_state = get_base_model_state_dict(model)
        for key in base_state:
            assert "lora" not in key.lower()
            assert "adapter" not in key.lower()
