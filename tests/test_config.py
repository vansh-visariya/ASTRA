"""
Unit tests for config: load_config, _deep_merge, environment variable overrides,
and default values.
"""

import os
from pathlib import Path
from unittest.mock import patch, mock_open

from astra.core.config import load_config, _deep_merge, DEFAULT_CONFIG


class TestDeepMerge:
    def test_shallow_merge(self):
        base = {"a": 1, "b": 2}
        override = {"b": 99, "c": 3}
        result = _deep_merge(base, override)
        assert result["a"] == 1
        assert result["b"] == 99
        assert result["c"] == 3

    def test_nested_merge(self):
        base = {"a": {"x": 1, "y": 2}}
        override = {"a": {"y": 99, "z": 3}}
        result = _deep_merge(base, override)
        assert result["a"]["x"] == 1
        assert result["a"]["y"] == 99
        assert result["a"]["z"] == 3

    def test_empty_override_no_change(self):
        base = {"a": 1, "b": 2}
        override = {}
        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": 2}

    def test_deep_nested(self):
        base = {"level1": {"level2": {"a": 1}}}
        override = {"level1": {"level2": {"a": 99}}}
        result = _deep_merge(base, override)
        assert result["level1"]["level2"]["a"] == 99


class TestLoadConfig:
    def test_defaults_only(self):
        with patch("astra.core.config._find_config_yaml", return_value=None):
            config = load_config(None)
            assert config["seed"] == 42
            assert "hf" in config["model"]
            assert config["client"]["num_clients"] == 20

    def test_unknown_key_path_returns_defaults(self):
        with patch("astra.core.config._find_config_yaml", return_value=None):
            config = load_config(None)
            assert config["seed"] == 42

    def test_default_structure_complete(self):
        with patch("astra.core.config._find_config_yaml", return_value=None):
            config = load_config(None)
            expected_keys = [
                "seed", "dataset", "model", "client", "server",
                "robust", "trust", "malicious", "privacy",
                "communication", "training",
            ]
            for key in expected_keys:
                assert key in config

    def test_env_override_seed(self):
        with patch("astra.core.config._find_config_yaml", return_value=None), \
             patch.dict(os.environ, {"ASTRA_SEED": "123"}, clear=True):
            config = load_config(None)
            assert config["seed"] == 123

    def test_env_override_secret_key(self):
        with patch("astra.core.config._find_config_yaml", return_value=None), \
             patch.dict(os.environ, {"SECRET_KEY": "my-secret"}, clear=True):
            config = load_config(None)
            assert config.get("secret_key") == "my-secret"

    def test_env_override_db_path(self):
        with patch("astra.core.config._find_config_yaml", return_value=None), \
             patch.dict(os.environ, {"DB_PATH": "/custom/astra.db"}, clear=True):
            config = load_config(None)
            assert config.get("db_path") == "/custom/astra.db"

    def test_merged_config_overrides(self):
        base = {"a": 1, "b": {"x": 1}}
        override = {"b": {"x": 99}}
        merged = _deep_merge(base, override)
        assert merged["b"]["x"] == 99
        assert merged["a"] == 1


class TestDefaultConfig:
    def test_model_defaults(self):
        assert "hf" in DEFAULT_CONFIG["model"]
        assert "hf_model_name" in DEFAULT_CONFIG["model"]["hf"]

    def test_server_defaults(self):
        assert DEFAULT_CONFIG["server"]["aggregator_window"] == 10
        assert DEFAULT_CONFIG["server"]["adaptive_lr"] is True

    def test_trust_defaults(self):
        assert DEFAULT_CONFIG["trust"]["init"] == 1.0
        assert DEFAULT_CONFIG["trust"]["quarantine_threshold"] == 0.35

    def test_communication_defaults(self):
        assert DEFAULT_CONFIG["communication"]["compression"] in ("none", "topk")
        assert 0.0 < DEFAULT_CONFIG["communication"]["topk_ratio"] <= 1.0

    def test_training_defaults(self):
        assert isinstance(DEFAULT_CONFIG["training"]["total_steps"], int)
        assert isinstance(DEFAULT_CONFIG["training"]["eval_interval_steps"], int)
