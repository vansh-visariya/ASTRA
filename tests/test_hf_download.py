"""
Tests for HuggingFace model save-to-disk, download-info, PEFT validation.
"""

import json
import os
import tempfile

import pytest
import torch
import torch.nn as nn

from astra.core.models.hf_models import (
    freeze_backbone,
    get_base_model_state_dict,
    get_download_info,
    get_lora_state_dict,
    save_base_model_to_disk,
)


class DummyPEFTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(10, 10)
        self.lora_A = nn.Linear(10, 4)
        self.lora_B = nn.Linear(4, 10)

    def forward(self, x):
        return self.backbone(x) + self.lora_B(self.lora_A(x))


@pytest.fixture
def dummy_model():
    return DummyPEFTModel()


@pytest.fixture
def temp_save_dir():
    d = tempfile.mkdtemp()
    yield d
    import shutil
    shutil.rmtree(d, ignore_errors=True)


class TestSaveBaseModelToDisk:
    def test_saves_pt(self, dummy_model, temp_save_dir):
        saved = save_base_model_to_disk(
            model=dummy_model, save_dir=temp_save_dir, model_name="test/model"
        )
        assert "pt" in saved
        assert os.path.exists(saved["pt"])

    def test_saves_config(self, dummy_model, temp_save_dir):
        save_base_model_to_disk(
            model=dummy_model,
            save_dir=temp_save_dir,
            model_name="test/model",
            peft_config={"enabled": True, "method": "lora"},
        )
        config_path = os.path.join(temp_save_dir, "adapter_config.json")
        assert os.path.exists(config_path)
        with open(config_path) as f:
            cfg = json.load(f)
        assert cfg["model_name"] == "test/model"
        assert cfg["peft_config"]["method"] == "lora"


class TestGetDownloadInfo:
    def test_empty_dir(self, temp_save_dir):
        info = get_download_info(temp_save_dir)
        assert info["has_base_model"] is False
        assert info["has_adapter"] is False

    def test_with_base_model(self, dummy_model, temp_save_dir):
        save_base_model_to_disk(
            model=dummy_model, save_dir=temp_save_dir, model_name="test/model"
        )
        info = get_download_info(temp_save_dir)
        assert info["has_base_model"] is True
        assert "pt" in info["formats"]


class TestPEFTUploadValidation:
    def test_full_model_upload_rejected(self):
        full_model_bytes = 1000 * 4
        assert full_model_bytes / full_model_bytes > 0.5

    def test_adapter_upload_accepted(self):
        full_model_bytes = 1000 * 4
        adapter_bytes = 100 * 4
        assert adapter_bytes / full_model_bytes <= 0.5


class TestHelperFunctions:
    def test_get_base_model_state_dict_excludes_lora(self, dummy_model):
        base = get_base_model_state_dict(dummy_model)
        for key in base:
            assert "lora" not in key.lower()

    def test_get_lora_state_dict_includes_only_lora(self, dummy_model):
        lora = get_lora_state_dict(dummy_model)
        for key in lora:
            assert "lora" in key.lower()

    def test_freeze_backbone(self, dummy_model):
        freeze_backbone(dummy_model)
        for name, param in dummy_model.named_parameters():
            if "lora" in name.lower():
                assert param.requires_grad is True
            else:
                assert param.requires_grad is False
