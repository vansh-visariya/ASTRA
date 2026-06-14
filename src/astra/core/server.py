"""
Async Federated Learning Server.

Implements an asynchronous server that processes client updates immediately
upon arrival without global synchronization barriers.

References:
- Xie et al., "Asynchronous Federated Optimization"
"""

import logging
import queue
import threading
import time
from collections import deque
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from astra.core.aggregation.aggregator import Aggregator
from astra.core.privacy.privacy import clip_and_noise
from astra.core.trust_manager import TrustManager
from astra.core.utils.metrics import compute_accuracy, compute_loss


class AsyncServer:
    """Asynchronous federated learning server."""

    def __init__(
        self,
        model: nn.Module,
        aggregator: Aggregator,
        config: dict[str, Any],
        val_loader: DataLoader | None = None,
    ):
        self.model = model
        self.aggregator = aggregator
        self.config = config
        self.val_loader = val_loader
        self.logger = logging.getLogger(__name__)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        self.is_peft = config.get("peft", {}).get("enabled", False)
        if self.is_peft:
            from astra.core.models.hf_models import freeze_backbone as _freeze

            _freeze(self.model)
            self.logger.info("PEFT mode enabled — backbone frozen, only LoRA params trainable")

        self.logger.info(
            "AsyncServer init: device=%s, peft=%s, window=%s, aggregator=%s",
            self.device,
            self.is_peft,
            config["server"]["aggregator_window"],
            aggregator.__class__.__name__,
        )

        self.global_version = 0
        self.running_global_estimate: np.ndarray | None = None

        self.aggregator_buffer: deque = deque(maxlen=config["server"]["aggregator_window"])
        self.running_momentum: np.ndarray | None = None

        self.trust_manager = TrustManager(config)

        self.update_queue: queue.Queue = queue.Queue()
        self.running = False
        self.lock = threading.Lock()

        self.current_lr = config["server"]["server_lr"]

        self._init_optimizer()

    def _init_optimizer(self):
        """Initialize server optimizer."""
        if self.config["server"]["optimizer"].lower() == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.config["server"]["server_lr"],
                momentum=self.config["server"].get("momentum", 0.9),
            )
        else:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.config["server"]["server_lr"]
            )
        self.logger.debug(
            "Server optimizer init: %s, lr=%s",
            self.optimizer.__class__.__name__,
            self.current_lr,
        )

    def start(self) -> None:
        """Start the server (simplified for demo)."""
        self.running = True
        self.logger.info("AsyncServer started (v%s)", self.global_version)

    def stop(self) -> None:
        """Stop the server."""
        self.running = False
        self.logger.info("AsyncServer stopped at v%s", self.global_version)

    def handle_update(self, client_update: dict[str, Any]) -> None:
        """Process a single client update immediately."""
        client_id = client_update.get("client_id", "unknown")
        staleness = self.global_version - client_update.get("client_version", 0)
        staleness_weight = np.exp(-self.config["server"]["async_lambda"] * staleness)

        self.logger.debug(
            "Received update: client=%s, staleness=%s, staleness_weight=%.4f, dataset_size=%s",
            client_id,
            staleness,
            staleness_weight,
            client_update.get("local_dataset_size", 1),
        )

        with self.lock:
            update_vector = self._decode_update(client_update.get("local_updates"))

            dp_enabled = self.config["privacy"]["dp_enabled"]
            dp_server = self.config["privacy"]["dp_mode"] == "server"
            if dp_enabled and dp_server:
                update_vector = clip_and_noise(
                    update_vector,
                    self.config["privacy"]["clip_norm"],
                    self.config["privacy"]["sigma"],
                )
                self.logger.debug("Server-side DP applied: clip=%.2f sigma=%.2f",
                                  self.config["privacy"]["clip_norm"],
                                  self.config["privacy"]["sigma"])

            trust_score = self.trust_manager.update_trust(
                client_id, update_vector, self.running_global_estimate
            )

            self.aggregator_buffer.append(
                {
                    "client_id": client_id,
                    "delta": update_vector,
                    "staleness_weight": staleness_weight,
                    "trust": trust_score,
                    "timestamp": client_update.get("timestamp", time.time()),
                    "local_dataset_size": client_update.get("local_dataset_size", 1),
                }
            )

            self.logger.debug(
                "Buffer: %s/%s — trust=%.3f",
                len(self.aggregator_buffer),
                self.aggregator_buffer.maxlen,
                trust_score,
            )

            self._maybe_aggregate()

    def _decode_update(self, encoded_update: Any) -> np.ndarray:
        """Decode client update from transport format."""
        if encoded_update is None:
            self.logger.warning("Received null update, returning zero array")
            return np.array([], dtype=np.float32)

        if isinstance(encoded_update, bytes):
            return np.frombuffer(encoded_update, dtype=np.float32)
        elif isinstance(encoded_update, np.ndarray):
            return encoded_update
        else:
            return np.array(encoded_update, dtype=np.float32)

    def _maybe_aggregate(self) -> bool:
        """Check if aggregation should occur and execute if ready."""
        min_window = self.config["server"]["aggregator_window"]

        if len(self.aggregator_buffer) >= min_window:
            self._perform_aggregation()
            return True
        return False

    def _perform_aggregation(self) -> None:
        """Execute robust aggregation and update global model."""
        buffer_list = list(self.aggregator_buffer)
        self.logger.info(
            "Aggregating %s updates (v%s -> v%s) with %s",
            len(buffer_list),
            self.global_version,
            self.global_version + 1,
            self.aggregator.__class__.__name__,
        )

        aggregated_delta = self.aggregator.aggregate(buffer_list)
        self.logger.debug("Aggregated delta shape=%s norm=%.4f", aggregated_delta.shape,
                          float(np.linalg.norm(aggregated_delta)))

        if self.running_momentum is None:
            self.running_momentum = aggregated_delta
        else:
            momentum = self.config["server"].get("momentum", 0.9)
            self.running_momentum = momentum * self.running_momentum + aggregated_delta

        self._apply_update(self.running_momentum)

        self.global_version += 1
        self.logger.info("Global model updated to v%s (mode=%s)", self.global_version,
                         "peft" if self.is_peft else "full")

        if self.running_global_estimate is None:
            self.running_global_estimate = aggregated_delta
        else:
            self.running_global_estimate = (
                0.9 * self.running_global_estimate + 0.1 * aggregated_delta
            )

        self._check_adaptive_lr(buffer_list)

        self.aggregator_buffer.clear()

    def _apply_update(self, delta: np.ndarray) -> None:
        """Apply aggregated delta to model parameters.

        In PEFT mode, only LoRA/adapter parameters are updated.
        """
        if self.is_peft:
            from astra.core.models.model_zoo import apply_peft_delta as _apply

            _apply(self.model, delta)
            return

        param_idx = 0
        for param in self.model.parameters():
            param_shape = param.shape
            param_size = np.prod(param_shape)

            if param_idx + param_size <= len(delta):
                param_data = delta[param_idx : param_idx + param_size].reshape(param_shape)
                param.data.add_(torch.from_numpy(param_data).float().to(param.device))
            param_idx += param_size

    def _check_adaptive_lr(self, buffer_snapshot: list[dict[str, Any]]) -> None:
        """Adaptively adjust learning rate based on instability."""
        if not self.config["server"].get("adaptive_lr", False):
            return

        recent_deltas = [b["delta"] for b in buffer_snapshot[-5:]]
        if len(recent_deltas) > 1:
            variance = np.var([np.linalg.norm(d) for d in recent_deltas])
            mean_norm = np.mean([np.linalg.norm(d) for d in recent_deltas])

            if mean_norm > 0 and variance / mean_norm > self.config["server"].get(
                "instability_threshold", 0.15
            ):
                self.current_lr *= self.config["server"].get("lr_decay_factor", 0.5)
                self.logger.warning(f"Instability detected. Reducing LR to {self.current_lr}")

    def evaluate(self) -> dict[str, float]:
        """Evaluate global model on validation set."""
        if self.val_loader is None:
            return {"accuracy": 0.0, "loss": 0.0}

        self.model.eval()
        accuracy = compute_accuracy(self.model, self.val_loader)
        loss = compute_loss(self.model, self.val_loader)

        return {
            "accuracy": accuracy,
            "loss": loss,
            "global_version": self.global_version,
            "trust_stats": str(self.trust_manager.get_stats()),  # type: ignore[dict-item]
        }

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "global_version": self.global_version,
            "config": self.config,
        }
        torch.save(checkpoint, path)
        self.logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location="cpu")
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.global_version = checkpoint["global_version"]
        self.logger.info(f"Checkpoint loaded from {path}")
