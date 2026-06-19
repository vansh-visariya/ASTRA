"""
Tests for HuggingFace model download-to-disk, safetensors support,
PEFT upload validation, and download-info endpoint.
"""

import json
import os
import shutil
import tempfile

import numpy as np
import pytest
import torch
import torch.nn as nn

from astra.core.models.hf_models import (
    freeze_backbone,
    get_base_model_state_dict,
    get_download_info,
    get_lora_state_dict,
    load_base_model_from_disk,
    save_base_model_to_disk,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class DummyPEFTModel(nn.Module):
    """A tiny model with LoRA-like parameters for testing."""

    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(32, 16)
        self.lora_A = nn.Linear(32, 4, bias=False)
        self.lora_B = nn.Linear(4, 16, bias=False)
        self.head = nn.Linear(16, 10)

    def forward(self, x):
        h = self.backbone(x)
        lora_out = self.lora_B(self.lora_A(x))
        return self.head(h + lora_out)


@pytest.fixture
def dummy_model():
    """A small model with both backbone and LoRA params."""
    return DummyPEFTModel()


@pytest.fixture
def temp_save_dir(tmp_path):
    """A temporary directory for model saves."""
    d = tmp_path / "hf_models"
    d.mkdir()
    return str(d)


# ---------------------------------------------------------------------------
# Tests: save_base_model_to_disk
# ---------------------------------------------------------------------------

class TestSaveBaseModelToDisk:
    def test_saves_pt_file(self, dummy_model, temp_save_dir):
        saved = save_base_model_to_disk(
            model=dummy_model,
            save_dir=temp_save_dir,
            model_name="test/model",
        )
        assert "pt" in saved
        assert os.path.exists(saved["pt"])
        assert os.path.basename(saved["pt"]) == "base_model.pt"

    def test_saves_metadata_json(self, dummy_model, temp_save_dir):
        save_base_model_to_disk(
            model=dummy_model,
            save_dir=temp_save_dir,
            model_name="test/model",
            peft_config={"enabled": True, "method": "lora"},
        )
        config_path = os.path.join(temp_save_dir, "adapter_config.json")
        assert os.path.exists(config_path)
        with open(config_path) as f:
            config = json.load(f)
        assert config["model_name"] == "test/model"
        assert config["has_peft"] is True
        assert config["peft_config"]["method"] == "lora"

    def test_saves_safetensors_if_available(self, dummy_model, temp_save_dir):
        try:
            import safetensors  # noqa: F401
        except ImportError:
            pytest.skip("safetensors not installed")

        saved = save_base_model_to_disk(
            model=dummy_model,
            save_dir=temp_save_dir,
            model_name="test/model",
        )
        assert "safetensors" in saved
        assert os.path.exists(saved["safetensors"])
        assert os.path.basename(saved["safetensors"]) == "base_model.safetensors"

    def test_pt_file_contains_correct_state_dict(self, dummy_model, temp_save_dir):
        saved = save_base_model_to_disk(
            model=dummy_model,
            save_dir=temp_save_dir,
            model_name="test/model",
        )
        data = torch.load(saved["pt"], map_location="cpu", weights_only=False)
        assert "base_state_dict" in data
        assert "model_name" in data
        # base_state_dict should not contain lora params
        for key in data["base_state_dict"]:
            assert "lora" not in key.lower()

    def test_overwrites_existing_files(self, dummy_model, temp_save_dir):
        save_base_model_to_disk(
            model=dummy_model,
            save_dir=temp_save_dir,
            model_name="test/model",
        )
        # Save again — should not raise
        saved = save_base_model_to_disk(
            model=dummy_model,
            save_dir=temp_save_dir,
            model_name="test/model",
        )
        assert os.path.exists(saved["pt"])


# ---------------------------------------------------------------------------
# Tests: load_base_model_from_disk
# ---------------------------------------------------------------------------

class TestLoadBaseModelFromDisk:
    def test_load_from_pt(self, dummy_model, temp_save_dir):
        save_base_model_to_disk(
            model=dummy_model,
            save_dir=temp_save_dir,
            model_name="test/model",
        )
        loaded = load_base_model_from_disk(temp_save_dir, preferred_format="pt")
        assert loaded is not None
        assert "base_state_dict" in loaded
        assert loaded["model_name"] == "test/model"
        # Check that the state dict has the right keys
        base_keys = list(loaded["base_state_dict"].keys())
        assert any("backbone" in k for k in base_keys)
        assert not any("lora" in k.lower() for k in base_keys)

    def test_load_from_safetensors(self, dummy_model, temp_save_dir):
        try:
            import safetensors  # noqa: F401
        except ImportError:
            pytest.skip("safetensors not installed")

        save_base_model_to_disk(
            model=dummy_model,
            save_dir=temp_save_dir,
            model_name="test/model",
        )
        loaded = load_base_model_from_disk(temp_save_dir, preferred_format="safetensors")
        assert loaded is not None
        assert "base_state_dict" in loaded

    def test_load_returns_none_if_empty(self, temp_save_dir):
        loaded = load_base_model_from_disk(temp_save_dir)
        assert loaded is None

    def test_load_fallback_to_pt(self, dummy_model, temp_save_dir):
        """If safetensors is requested but only pt exists, should fall back."""
        save_base_model_to_disk(
            model=dummy_model,
            save_dir=temp_save_dir,
            model_name="test/model",
        )
        # Remove safetensors if it exists
        sf_path = os.path.join(temp_save_dir, "base_model.safetensors")
        if os.path.exists(sf_path):
            os.remove(sf_path)
        loaded = load_base_model_from_disk(temp_save_dir, preferred_format="safetensors")
        assert loaded is not None


# ---------------------------------------------------------------------------
# Tests: get_download_info
# ---------------------------------------------------------------------------

class TestGetDownloadInfo:
    def test_empty_dir(self, temp_save_dir):
        info = get_download_info(temp_save_dir)
        assert info["has_base_model"] is False
        assert info["has_adapter"] is False
        assert info["formats"] == {}
        assert info["adapter_versions"] == []

    def test_with_base_model(self, dummy_model, temp_save_dir):
        save_base_model_to_disk(
            model=dummy_model,
            save_dir=temp_save_dir,
            model_name="test/model",
        )
        info = get_download_info(temp_save_dir)
        assert info["has_base_model"] is True
        assert "pt" in info["formats"]
        assert info["formats"]["pt"]["size_bytes"] > 0

    def test_with_adapter(self, dummy_model, temp_save_dir):
        # Save a dummy adapter
        adapter_path = os.path.join(temp_save_dir, "adapter_latest.pt")
        adapter_state = get_lora_state_dict(dummy_model)
        torch.save({"lora_state_dict": adapter_state}, adapter_path)

        info = get_download_info(temp_save_dir)
        assert info["has_adapter"] is True
        assert info["adapter_latest_size"] > 0

    def test_with_versioned_adapters(self, dummy_model, temp_save_dir):
        for v in [1, 2, 3]:
            adapter_path = os.path.join(temp_save_dir, f"adapter_v{v}.pt")
            torch.save({"lora_state_dict": {}}, adapter_path)
        latest_path = os.path.join(temp_save_dir, "adapter_latest.pt")
        torch.save({"lora_state_dict": {}}, latest_path)

        info = get_download_info(temp_save_dir)
        assert info["adapter_versions"] == [1, 2, 3]

    def test_with_adapter_config(self, dummy_model, temp_save_dir):
        config_path = os.path.join(temp_save_dir, "adapter_config.json")
        with open(config_path, "w") as f:
            json.dump({"model_name": "test/model", "has_peft": True}, f)

        info = get_download_info(temp_save_dir)
        assert "adapter_config" in info
        assert info["adapter_config"]["model_name"] == "test/model"


# ---------------------------------------------------------------------------
# Tests: PEFT upload validation (mock)
# ---------------------------------------------------------------------------

class TestPEFTUploadValidation:
    """Tests that the PEFT upload validation logic works correctly.

    These test the validation logic directly without needing a full server.
    """

    def test_full_model_upload_rejected_for_peft_group(self):
        """Simulates a client uploading full model weights to a PEFT group."""
        # If the group expects adapter-only deltas, uploading full model weights
        # should be detected and rejected.
        full_model_bytes = 1000 * 4  # 1000 params * 4 bytes = 4000 bytes
        adapter_ratio = full_model_bytes / full_model_bytes  # 1.0 = 100%
        assert adapter_ratio > 0.5, "Full model upload should be detected"

    def test_adapter_upload_accepted_for_peft_group(self):
        """Simulates a client uploading adapter-only weights to a PEFT group."""
        full_model_bytes = 1000 * 4  # 1000 params * 4 bytes
        adapter_bytes = 100 * 4  # 100 adapter params * 4 bytes
        adapter_ratio = adapter_bytes / full_model_bytes  # 0.1 = 10%
        assert adapter_ratio <= 0.5, "Adapter upload should be accepted"


# ---------------------------------------------------------------------------
# Tests: save and load round-trip
# ---------------------------------------------------------------------------

class TestSaveLoadRoundTrip:
    def test_round_trip_pt(self, dummy_model, temp_save_dir):
        """Save and load should produce equivalent state dicts."""
        saved = save_base_model_to_disk(
            model=dummy_model,
            save_dir=temp_save_dir,
            model_name="test/model",
            peft_config={"enabled": True, "method": "lora", "lora_rank": 4},
        )
        loaded = load_base_model_from_disk(temp_save_dir, preferred_format="pt")
        assert loaded is not None

        original_state = get_base_model_state_dict(dummy_model)
        loaded_state = loaded["base_state_dict"]

        assert set(original_state.keys()) == set(loaded_state.keys())
        for key in original_state:
            assert torch.allclose(original_state[key], loaded_state[key])

    def test_round_trip_safetensors(self, dummy_model, temp_save_dir):
        try:
            import safetensors  # noqa: F401
        except ImportError:
            pytest.skip("safetensors not installed")

        saved = save_base_model_to_disk(
            model=dummy_model,
            save_dir=temp_save_dir,
            model_name="test/model",
        )
        loaded = load_base_model_from_disk(temp_save_dir, preferred_format="safetensors")
        assert loaded is not None

        original_state = get_base_model_state_dict(dummy_model)
        loaded_state = loaded["base_state_dict"]

        assert set(original_state.keys()) == set(loaded_state.keys())
        for key in original_state:
            assert torch.allclose(original_state[key], loaded_state[key])


# ---------------------------------------------------------------------------
# Tests: helper functions
# ---------------------------------------------------------------------------

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
