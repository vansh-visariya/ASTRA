"""
Model utilities for federated learning.

Parameter serialization, PEFT delta handling, and flat delta operations
shared between clients and the server.
"""

import numpy as np
import torch
import torch.nn as nn


class SimpleMLP(nn.Module):
    """Simple MLP for basic experiments."""

    def __init__(self, input_dim: int = 784, num_classes: int = 10, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


def _is_lora_param(name: str) -> bool:
    """Check if a parameter name belongs to a LoRA/adapter module."""
    return "lora" in name.lower() or "adapter" in name.lower()


def flatten_peft_params(model: nn.Module) -> np.ndarray:
    """Flatten only LoRA/adapter parameters in sorted-name order."""
    peft_params = sorted(
        [(name, param) for name, param in model.named_parameters() if _is_lora_param(name)],
        key=lambda x: x[0],
    )
    if not peft_params:
        return np.array([], dtype=np.float32)
    return np.concatenate(
        [param.data.cpu().numpy().flatten().astype(np.float32) for _, param in peft_params]
    )


def apply_peft_delta(model: nn.Module, flat_delta: np.ndarray) -> None:
    """Apply a flat LoRA delta to the model's LoRA parameters in-place."""
    offset = 0
    for _name, param in sorted(
        [(n, p) for n, p in model.named_parameters() if _is_lora_param(n)],
        key=lambda x: x[0],
    ):
        size = param.numel()
        if offset + size <= len(flat_delta):
            delta_slice = flat_delta[offset : offset + size].reshape(param.shape)
            param.data.add_(torch.from_numpy(delta_slice).float().to(param.device))
        offset += size


def flatten_all_params(model: nn.Module) -> np.ndarray:
    """Flatten ALL model parameters in deterministic sorted-name order."""
    params = sorted(
        [(name, param) for name, param in model.named_parameters()],
        key=lambda x: x[0],
    )
    if not params:
        return np.array([], dtype=np.float32)
    return np.concatenate(
        [param.data.cpu().numpy().flatten().astype(np.float32) for _, param in params]
    )


def apply_flat_delta(model: nn.Module, flat_delta: np.ndarray) -> None:
    """Apply a flat delta to ALL model parameters in sorted-name order."""
    offset = 0
    for _name, param in sorted(
        [(n, p) for n, p in model.named_parameters()],
        key=lambda x: x[0],
    ):
        size = param.numel()
        if offset + size <= len(flat_delta):
            delta_slice = flat_delta[offset : offset + size].reshape(param.shape)
            param.data.add_(torch.from_numpy(delta_slice).float().to(param.device))
        offset += size
