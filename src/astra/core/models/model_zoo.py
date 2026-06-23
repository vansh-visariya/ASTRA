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
    """Apply a flat LoRA delta to the model's LoRA parameters in-place.

    Raises ValueError if the delta has more elements than the model's
    LoRA parameters (indicates a PEFT config mismatch between client and
    server — e.g. client trained with ``target_modules='all-linear'`` but
    server only has LoRA on ``['q_proj', 'v_proj']``).
    """
    lora_params = sorted(
        [(n, p) for n, p in model.named_parameters() if _is_lora_param(n)],
        key=lambda x: x[0],
    )
    total_lora_params = sum(p.numel() for _, p in lora_params)

    if len(flat_delta) != total_lora_params:
        raise ValueError(
            f"PEFT delta size mismatch: delta has {len(flat_delta):,} elements "
            f"but the model's LoRA parameters total {total_lora_params:,}. "
            f"This usually means the client's LoRA config (target_modules, "
            f"rank) doesn't match the server's. "
            f"Delta params: {len(flat_delta):,} | Model LoRA params: {total_lora_params:,}"
        )

    offset = 0
    for _name, param in lora_params:
        size = param.numel()
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
    """Apply a flat delta to ALL model parameters in sorted-name order.

    Raises ValueError if the delta size doesn't match the model's total
    parameter count.
    """
    all_params = sorted(
        [(n, p) for n, p in model.named_parameters()],
        key=lambda x: x[0],
    )
    total_params = sum(p.numel() for _, p in all_params)

    if len(flat_delta) != total_params:
        raise ValueError(
            f"Delta size mismatch: delta has {len(flat_delta):,} elements "
            f"but the model has {total_params:,} parameters. "
            f"The uploaded weights don't match the group's model architecture."
        )

    offset = 0
    for _name, param in all_params:
        size = param.numel()
        delta_slice = flat_delta[offset : offset + size].reshape(param.shape)
        param.data.add_(torch.from_numpy(delta_slice).float().to(param.device))
        offset += size
