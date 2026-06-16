"""
Model Registry for managing baseline models.

Supports:
- HuggingFace models (with PEFT)
- External model architectures via import path
- Local model files

References:
- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models"
"""

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class ModelInfo:
    """Model metadata container."""

    def __init__(
        self,
        model_id: str,
        model_type: str,
        architecture: str,
        total_params: int,
        trainable_params: int,
        is_peft: bool = False,
        peft_method: str | None = None,
        source: str = "local",
        model_path: str | None = None,
        config: dict | None = None,
    ):
        self.model_id = model_id
        self.model_type = model_type
        self.architecture = architecture
        self.total_params = total_params
        self.trainable_params = trainable_params
        self.is_peft = is_peft
        self.peft_method = peft_method
        self.source = source
        self.model_path = model_path
        self.config = config or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_type": self.model_type,
            "architecture": self.architecture,
            "total_params": self.total_params,
            "trainable_params": self.trainable_params,
            "is_peft": self.is_peft,
            "peft_method": self.peft_method,
            "source": self.source,
            "model_path": self.model_path,
            "config": self.config,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelInfo":
        return cls(**data)


class ModelRegistry:
    """Central registry for managing baseline models."""

    def __init__(self, cache_dir: str = "./models"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.models: dict[str, ModelInfo] = {}
        self.model_instances: dict[str, nn.Module] = {}
        self.model_factories: dict[str, Callable[[], nn.Module]] = {}

        self.logger = logging.getLogger(__name__)

    def register_factory(
        self, model_id: str, factory: Callable[[], nn.Module], model_info: ModelInfo
    ) -> None:
        """Register a model with its factory function."""
        self.models[model_id] = model_info
        self.model_factories[model_id] = factory
        self.logger.info(
            "Registered factory for '%s' (%s, %s params)",
            model_id,
            model_info.architecture,
            model_info.total_params,
        )

    def build_model(self, model_id: str, device: str = "cpu") -> nn.Module:
        """Instantiate a model from the registry by its ID."""
        if model_id in self.model_instances:
            return self.model_instances[model_id].to(device)

        if model_id in self.model_factories:
            model = self.model_factories[model_id]()
            model = model.to(device)
            self.model_instances[model_id] = model
            return model

        raise ValueError(
            f"No factory registered for model '{model_id}'. "
            f"Available: {list(self.model_factories.keys())}"
        )

    def register_hf_model(
        self, model_name: str, use_peft: bool = True, peft_config: dict | None = None
    ) -> ModelInfo:
        """Register a HuggingFace model."""
        from astra.core.models.hf_models import load_hf_peft_model

        model_id = f"hf_{model_name.replace('/', '_')}"
        if use_peft:
            model_id += "_peft"

        if model_id in self.models:
            self.logger.info(f"Model {model_id} already registered")
            return self.models[model_id]

        self.logger.info(f"Loading HF model: {model_name}")

        try:
            peft_cfg = peft_config or {
                "enabled": use_peft,
                "method": "lora",
                "lora_rank": 8,
                "lora_alpha": 16,
                "target_modules": ["q_proj", "v_proj"],
            }

            model, processor = load_hf_peft_model(model_name, peft_cfg, device="cpu")

            total_params = sum(p.numel() for p in model.parameters())
            trainable = (
                sum(p.numel() for p in model.parameters() if p.requires_grad)
                if use_peft
                else total_params
            )

            model_type = "vision"
            if "text" in model_name.lower() or "bert" in model_name.lower() or "gpt" in model_name.lower():
                model_type = "text"
            if "clip" in model_name.lower() or "blip" in model_name.lower():
                model_type = "multimodal"

            model_info = ModelInfo(
                model_id=model_id,
                model_type=model_type,
                architecture=model_name,
                total_params=total_params,
                trainable_params=trainable,
                is_peft=use_peft,
                peft_method=peft_cfg.get("method") if use_peft else None,
                source="huggingface",
                model_path=model_name,
                config=peft_cfg,
            )

            self.models[model_id] = model_info
            self.model_instances[model_id] = model
            self.model_factories[model_id] = lambda m=model_name, c=peft_cfg: load_hf_peft_model(
                m, c, device="cpu"
            )[0]

            self.logger.info(f"Registered HF model: {model_id} ({total_params:,} params)")
            return model_info

        except Exception as e:
            self.logger.error(f"Failed to load HF model {model_name}: {e}")
            raise

    def register_local_model(
        self, model_id: str, model_path: str, architecture: str = "Custom"
    ) -> ModelInfo:
        """Register a local .pt model file."""
        if model_id in self.models:
            return self.models[model_id]

        try:
            state_dict = torch.load(model_path, map_location="cpu")
            total_params = sum(p.numel() for p in state_dict.values())

            model_info = ModelInfo(
                model_id=model_id,
                model_type="custom",
                architecture=architecture,
                total_params=total_params,
                trainable_params=total_params,
                is_peft=False,
                source="local",
                model_path=model_path,
            )

            self.models[model_id] = model_info
            self.logger.info(f"Registered local model: {model_id}")
            return model_info

        except Exception as e:
            self.logger.error(f"Failed to load local model: {e}")
            raise

    def load_model(self, model_id: str, device: str = "cpu") -> nn.Module:
        """Load model instance."""
        try:
            return self.build_model(model_id, device=device)
        except ValueError:
            pass

        if model_id in self.model_instances:
            model = self.model_instances[model_id]
            return model.to(device)

        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found in registry")

        model_info = self.models[model_id]

        if model_info.source == "huggingface":
            from astra.core.models.hf_models import load_hf_peft_model

            model_path = model_info.model_path or ""
            model, _ = load_hf_peft_model(model_path, model_info.config, device=device)
        elif model_info.source == "local":
            model = torch.load(model_info.model_path or "", map_location=device)
        else:
            raise ValueError(f"No loader for source '{model_info.source}' on model '{model_id}'")

        self.model_instances[model_id] = model
        return model

    def list_models(self, model_type: str | None = None) -> list[dict[str, Any]]:
        """List all registered models."""
        models = list(self.models.values())
        if model_type:
            models = [m for m in models if m.model_type == model_type]
        return [m.to_dict() for m in models]

    def get_model_info(self, model_id: str) -> dict[str, Any] | None:
        """Get model info by ID."""
        if model_id in self.models:
            return self.models[model_id].to_dict()
        return None

    def validate_model(self, model_id: str) -> tuple[bool, str]:
        """Validate model compatibility."""
        if model_id not in self.models:
            return False, f"Model {model_id} not found"
        model_info = self.models[model_id]
        if model_info.total_params > 1_000_000_000:
            return False, "Model exceeds 1B parameters"
        return True, "Valid"

    def save_registry(self, path: str) -> None:
        """Save registry to JSON file."""
        data = {model_id: info.to_dict() for model_id, info in self.models.items()}
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load_registry(self, path: str) -> None:
        """Load registry from JSON file."""
        with open(path) as f:
            data = json.load(f)
        for model_id, info_dict in data.items():
            self.models[model_id] = ModelInfo.from_dict(info_dict)


# Global registry instance
_global_registry: ModelRegistry | None = None


def get_registry() -> ModelRegistry:
    """Get global model registry instance."""
    global _global_registry
    if _global_registry is None:
        _global_registry = ModelRegistry()
    return _global_registry
