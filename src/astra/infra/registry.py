"""
Model Registry for managing baseline and registered models.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn

__all__ = ["ModelInfo", "ModelRegistry", "get_registry"]


@dataclass
class ModelInfo:
    """Metadata for a registered model."""

    model_id: str
    model_type: str
    architecture: str
    total_params: int = 0
    trainable_params: int = 0
    is_peft: bool = False
    peft_method: str | None = None
    source: str = "registry"
    model_path: str | None = None
    config: dict[str, Any] = field(default_factory=dict)

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

            model, info = load_hf_peft_model(model_name, peft_cfg)
            total_params = info.get("total_params", 0)
            trainable_params = info.get("trainable_params", 0)

            model_info = ModelInfo(
                model_id=model_id,
                model_type="huggingface",
                architecture=model_name,
                total_params=total_params,
                trainable_params=trainable_params,
                is_peft=use_peft,
                peft_method="lora" if use_peft else None,
                source="huggingface",
                model_path=model_name,
                config=peft_cfg,
            )

            self.models[model_id] = model_info
            self.model_instances[model_id] = model
            self.logger.info(f"Registered HF model: {model_id} ({total_params:,} params)")
            return model_info

        except Exception as e:
            self.logger.error(f"Failed to load HF model {model_name}: {e}")
            raise

    def list_models(self, model_type: str | None = None) -> list[dict[str, Any]]:
        """List all registered models."""
        models = list(self.models.values())
        if model_type:
            models = [m for m in models if m.model_type == model_type]
        return [m.to_dict() for m in models]

    def get_model_info(self, model_id: str) -> dict[str, Any] | None:
        """Get model info by ID."""
        if model_id not in self.models:
            return None
        return self.models[model_id].to_dict()

    def validate_model(self, model_id: str) -> tuple[bool, str]:
        """Validate model compatibility."""
        if model_id not in self.models:
            return False, f"Model {model_id} not found"
        model_info = self.models[model_id]
        if model_info.total_params > 1_000_000_000:
            return False, "Model exceeds 1B parameters"
        return True, "Valid"


# Global registry instance
_global_registry: ModelRegistry | None = None


def get_registry() -> ModelRegistry:
    """Get global model registry instance."""
    global _global_registry
    if _global_registry is None:
        _global_registry = ModelRegistry()
    return _global_registry
