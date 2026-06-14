"""
Federated Learning Client.

Implements local training on client devices and update generation.

References:
- McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data"
"""

import logging
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from astra.core.compression import topk_sparsify
from astra.core.models.hf_models import (  # noqa: F401
    freeze_backbone,
    get_lora_state_dict,
    load_lora_state_dict,
)
from astra.core.models.model_zoo import flatten_all_params, flatten_peft_params
from astra.core.privacy.malicious_simulator import MaliciousSimulator
from astra.core.privacy.privacy import clip_and_noise


class FLClient:
    """Federated learning client."""

    def __init__(
        self,
        client_id: str,
        train_data: Any,
        model_factory: Callable[[], nn.Module],
        config: dict[str, Any],
    ):
        self.client_id = client_id
        self.train_data = train_data
        self.model_factory = model_factory
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.model = model_factory()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        self.is_peft = config.get("peft", {}).get("enabled", False)
        if self.is_peft:
            freeze_backbone(self.model)
            self.logger.debug("Client %s: backbone frozen (PEFT mode)", self.client_id)

        self.client_version = 0

        self.malicious_simulator = MaliciousSimulator(config)
        self.is_malicious = self._check_if_malicious()

        self.logger.info(
            "FLClient %s init: device=%s, peft=%s, malicious=%s, params=%s",
            self.client_id,
            self.device,
            self.is_peft,
            self.is_malicious,
            sum(p.numel() for p in self.model.parameters() if p.requires_grad),
        )

        self._init_optimizer()
        self._init_data_loader()

    def _check_if_malicious(self) -> bool:
        """Determine if this client is malicious based on config."""
        malicious_ratio = self.config["malicious"].get("ratio", 0)
        if malicious_ratio == 0:
            return False

        client_hash = hash(self.client_id)
        threshold = int(1 / malicious_ratio)
        return client_hash % threshold == 0

    def _init_optimizer(self):
        """Initialize client optimizer (only trainable params)."""
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.config["client"]["lr"],
            weight_decay=self.config["client"].get("weight_decay", 0.0),
        )

    def _init_data_loader(self):
        """Initialize data loader."""
        batch_size = self.config["client"]["batch_size"]

        if isinstance(self.train_data, Subset):
            self.train_loader = DataLoader(
                self.train_data, batch_size=batch_size, shuffle=True, num_workers=0
            )
        else:
            self.train_loader = DataLoader(
                self.train_data, batch_size=batch_size, shuffle=True, num_workers=0
            )

    def local_train(self) -> dict[str, Any]:
        """
        Run local training on client data.

        In PEFT mode, only LoRA adapter parameters are trained and
        transmitted. In non-PEFT mode, full model weights are used.

        Returns:
            Dictionary containing client update for server.
        """
        self.model.train()

        local_epochs = self.config["client"]["local_epochs"]
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        epoch_metrics = []

        initial_weights = self._get_weights()

        for epoch in range(local_epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_samples = 0
            for _batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()

                output = self.model(data)
                loss = nn.functional.cross_entropy(output, target)
                loss.backward()
                self.optimizer.step()

                batch_size = len(target)
                total_loss += loss.item() * batch_size
                pred = output.argmax(dim=1)
                total_correct += (pred == target).sum().item()
                total_samples += batch_size

                epoch_loss += loss.item() * batch_size
                epoch_correct += (pred == target).sum().item()
                epoch_samples += batch_size

            epoch_loss_avg = epoch_loss / epoch_samples if epoch_samples > 0 else 0.0
            epoch_accuracy = epoch_correct / epoch_samples if epoch_samples > 0 else 0.0
            epoch_metrics.append(
                {
                    "epoch": epoch + 1,
                    "loss": epoch_loss_avg,
                    "accuracy": epoch_accuracy,
                    "samples": epoch_samples,
                }
            )
            self.logger.info(
                "Client %s epoch %s/%s: loss=%.4f, acc=%.4f",
                self.client_id,
                epoch + 1,
                local_epochs,
                epoch_loss_avg,
                epoch_accuracy,
            )

        final_weights = self._get_weights()
        weight_delta = self._compute_weight_delta(initial_weights, final_weights)

        self.logger.debug(
            "Client %s delta: shape=%s, norm=%.4f, min=%.4f, max=%.4f",
            self.client_id,
            weight_delta.shape,
            float(np.linalg.norm(weight_delta)),
            float(np.min(weight_delta)),
            float(np.max(weight_delta)),
        )

        if self.is_malicious:
            weight_delta = self.malicious_simulator.inject_attack(weight_delta, self.client_id)
            self.logger.warning("Client %s: malicious attack injected", self.client_id)

        if self.config["privacy"]["dp_enabled"] and self.config["privacy"]["dp_mode"] == "client":
            weight_delta = clip_and_noise(
                weight_delta, self.config["privacy"]["clip_norm"], self.config["privacy"]["sigma"]
            )
            self.logger.debug("Client %s: DP applied (client-side)", self.client_id)

        if self.config["communication"]["compression"] == "topk":
            k_ratio = self.config["communication"].get("topk_ratio", 0.1)
            weight_delta, compress_meta = topk_sparsify(weight_delta, k_ratio)
            self.logger.debug(
                "Client %s: top-k compression (ratio=%.2f, %s->%s elements)",
                self.client_id,
                k_ratio,
                compress_meta.get("original_size"),
                compress_meta.get("compressed_size"),
            )

        self.client_version += 1
        self.logger.info(
            "Client %s: update v%s ready — %.1f KB, loss=%.4f, acc=%.4f",
            self.client_id,
            self.client_version,
            weight_delta.nbytes / 1024,
            total_loss / max(total_samples, 1),
            total_correct / max(total_samples, 1) if total_samples > 0 else 0.0,
        )

        train_loss = total_loss / total_samples if total_samples > 0 else 0.0
        train_accuracy = total_correct / total_samples if total_samples > 0 else 0.0

        update = {
            "client_id": self.client_id,
            "client_version": self.client_version,
            "local_updates": weight_delta.tobytes(),
            "update_type": "delta",
            "local_dataset_size": len(self.train_data),
            "timestamp": time.time(),
            "meta": {
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "local_steps": local_epochs * len(self.train_loader),
                "epoch_metrics": epoch_metrics,
            },
        }

        return update

    def _get_weights(self) -> np.ndarray:
        """Get model weights as flattened numpy array.

        In PEFT mode, only LoRA/adapter params are returned.
        Falls back to all params in sorted-name order if no PEFT params are found.
        """
        if self.is_peft:
            peft_flat = flatten_peft_params(self.model)
            if len(peft_flat) > 0:
                return peft_flat
        return flatten_all_params(self.model)

    def get_adapter_state(self) -> dict[str, torch.Tensor]:
        """Get current LoRA adapter weights as named tensors (PEFT mode only)."""
        return get_lora_state_dict(self.model)

    def load_adapter_weights(self, adapter_state: dict[str, torch.Tensor]) -> None:
        """Load LoRA adapter weights from server (PEFT mode only)."""
        load_lora_state_dict(self.model, adapter_state)

    def _compute_weight_delta(self, initial: np.ndarray, final: np.ndarray) -> np.ndarray:
        """Compute weight delta (final - initial)."""
        return final - initial

    def send_update(self, server: Any) -> None:
        """Send update to server (simulated networking)."""
        update = self.local_train()
        server.handle_update(update)

    def reset_model(self) -> None:
        """Reset model to initial state."""
        self.model = self.model_factory()
        if self.is_peft:
            freeze_backbone(self.model)
        self._init_optimizer()

    def get_client_stats(self) -> dict[str, Any]:
        """Get client statistics."""
        return {
            "client_id": self.client_id,
            "is_malicious": self.is_malicious,
            "dataset_size": len(self.train_data),
            "client_version": self.client_version,
        }
