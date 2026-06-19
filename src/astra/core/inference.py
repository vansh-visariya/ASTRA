"""
Privacy-Preserving Inference Module for Federated Learning.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn


@dataclass
class InferenceResult:
    """Result of inference operation."""

    predictions: np.ndarray
    probabilities: np.ndarray | None
    confidence: float
    method: str
    metadata: dict[str, Any]


class InferenceModule:
    """Main inference module coordinating different inference methods."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.inference_methods: dict[str, Any] = {}
        self.default_method = "server_side"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def register_method(self, method_name: str, method: Any) -> None:
        self.inference_methods[method_name] = method
        self.logger.info(f"Registered inference method: {method_name}")

    def predict(self, input_data: Any, method: str | None = None, **kwargs) -> InferenceResult:
        method_name = method or self.default_method
        if method_name not in self.inference_methods:
            raise ValueError(f"Unknown method: {method_name}")
        return self.inference_methods[method_name].predict(input_data)

    def get_available_methods(self) -> list[str]:
        return list(self.inference_methods.keys())


def create_inference_module(
    config: dict[str, Any], model: nn.Module | None = None
) -> InferenceModule:
    """Create and configure inference module."""
    module = InferenceModule(config)
    if model:
        from dataclasses import dataclass as _dc

        @_dc
        class _ServerInfer:
            model: nn.Module
            device: torch.device

            def __post_init__(self):
                self.model.to(self.device)
                self.model.eval()

            def predict(self, input_data: Any) -> InferenceResult:
                with torch.no_grad():
                    tensor = (
                        torch.from_numpy(input_data)
                        if isinstance(input_data, np.ndarray)
                        else input_data
                    ).to(self.device)
                    if tensor.dim() == 3:
                        tensor = tensor.unsqueeze(0)
                    output = self.model(tensor)
                    probs = torch.softmax(output, dim=1)
                    return InferenceResult(
                        predictions=output.argmax(dim=1).cpu().numpy(),
                        probabilities=probs.cpu().numpy(),
                        confidence=probs.max().item(),
                        method="server_side",
                        metadata={"device": str(self.device)},
                    )

        module.register_method("server_side", _ServerInfer(model=model, device=module.device))
    return module
