"""
Privacy-Preserving Inference Module for Federated Learning.

Provides modular inference capabilities without exposing the full model:
1. Server-side inference (data leaves client, model stays)
2. Client-side inference (full model downloaded)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn


class InferenceMethod(ABC):
    """Abstract base class for inference methods."""

    @abstractmethod
    def predict(self, input_data: Any) -> InferenceResult:
        """Run inference and return results."""
        pass

    @abstractmethod
    def get_method_name(self) -> str:
        """Return the method name."""
        pass


@dataclass
class InferenceResult:
    """Result of inference operation."""

    predictions: np.ndarray
    probabilities: np.ndarray | None
    confidence: float
    method: str
    metadata: dict[str, Any]


class ServerSideInference(InferenceMethod):
    """Server-side inference where data is sent to server."""

    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, input_data: Any) -> InferenceResult:
        """Run inference on server."""
        with torch.no_grad():
            if isinstance(input_data, np.ndarray):
                tensor = torch.from_numpy(input_data)
            elif isinstance(input_data, torch.Tensor):
                tensor = input_data
            else:
                raise ValueError(f"Unsupported input type: {type(input_data)}")

            tensor = tensor.to(self.device)

            if tensor.dim() == 3:
                tensor = tensor.unsqueeze(0)

            output = self.model(tensor)
            probabilities = torch.softmax(output, dim=1)
            predictions = output.argmax(dim=1)
            confidence = probabilities.max().item()

            return InferenceResult(
                predictions=predictions.cpu().numpy(),
                probabilities=probabilities.cpu().numpy(),
                confidence=confidence,
                method="server_side",
                metadata={"device": str(self.device)},
            )

    def get_method_name(self) -> str:
        return "Server-Side Inference"


class ClientSideInference(InferenceMethod):
    """Full model inference on client side."""

    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, input_data: Any) -> InferenceResult:
        """Run full inference on client."""
        with torch.no_grad():
            if isinstance(input_data, np.ndarray):
                tensor = torch.from_numpy(input_data)
            elif isinstance(input_data, torch.Tensor):
                tensor = input_data
            else:
                raise ValueError(f"Unsupported input type: {type(input_data)}")

            tensor = tensor.to(self.device)

            if tensor.dim() == 3:
                tensor = tensor.unsqueeze(0)

            output = self.model(tensor)
            probabilities = torch.softmax(output, dim=1)
            predictions = output.argmax(dim=1)
            confidence = probabilities.max().item()

            return InferenceResult(
                predictions=predictions.cpu().numpy(),
                probabilities=probabilities.cpu().numpy(),
                confidence=confidence,
                method="client_side",
                metadata={"device": str(self.device)},
            )

    def get_method_name(self) -> str:
        return "Client-Side Inference"


class InferenceModule:
    """Main inference module coordinating different inference methods."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.inference_methods: dict[str, InferenceMethod] = {}
        self.default_method = "server_side"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def register_method(self, method_name: str, method: InferenceMethod) -> None:
        """Register an inference method."""
        self.inference_methods[method_name] = method
        self.logger.info(f"Registered inference method: {method_name}")

    def predict(self, input_data: Any, method: str | None = None, **kwargs) -> InferenceResult:
        """Run inference using specified or default method."""
        method_name = method or self.default_method

        if method_name not in self.inference_methods:
            raise ValueError(f"Unknown method: {method_name}")

        return self.inference_methods[method_name].predict(input_data)

    def create_server_side(self, model: nn.Module) -> ServerSideInference:
        """Create server-side inference instance."""
        method = ServerSideInference(model, device=str(self.device))
        self.register_method("server_side", method)
        return method

    def create_client_side(self, model: nn.Module) -> ClientSideInference:
        """Create client-side inference instance."""
        method = ClientSideInference(model, device=str(self.device))
        self.register_method("client_side", method)
        return method

    def get_available_methods(self) -> list[str]:
        """Get list of available inference methods."""
        return list(self.inference_methods.keys())


def create_inference_module(
    config: dict[str, Any], model: nn.Module | None = None
) -> InferenceModule:
    """Create and configure inference module."""
    module = InferenceModule(config)

    if model:
        module.create_server_side(model)
        module.create_client_side(model)

    return module
