"""
Async Federated Learning Server.

Pre-processes incoming client updates (DP, trust, buffer) and hands
the processed update back to the caller. The caller (route or
GroupManager) is responsible for actual aggregation and model update.

This avoids the double-aggregation bug where both AsyncServer and
GroupManager applied the same deltas to the shared model object.
"""

import logging
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


class AsyncServer:
    """Pre-processes client updates: DP, trust scoring, buffer management.

    This class does NOT aggregate or update the model — that is handled
    by GroupManager, which receives the processed (DP'd) update vector
    and aggregates per-group buffers.
    """

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

        self.lock = threading.Lock()

        self.current_lr = config["server"]["server_lr"]

        # Last aggregation results for the route to sync from
        self.last_aggregation_result: dict | None = None

    def handle_update(self, client_update: dict[str, Any]) -> dict[str, Any]:
        """Apply DP + trust scoring to a client update.

        The processed (DP'd) update vector is stored in
        ``client_update["local_updates"]`` as bytes. The caller should
        pass this same dict to GroupManager so both paths use the
        identical DP'd vector.

        Returns the (mutated) client_update dict for chaining.
        """
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

            # Server-side DP: clip + noise
            dp_enabled = self.config["privacy"]["dp_enabled"]
            dp_server = self.config["privacy"]["dp_mode"] == "server"
            if dp_enabled and dp_server:
                update_vector = clip_and_noise(
                    update_vector,
                    self.config["privacy"]["clip_norm"],
                    self.config["privacy"]["sigma"],
                )
                self.logger.debug(
                    "Server-side DP applied: clip=%.2f sigma=%.2f",
                    self.config["privacy"]["clip_norm"],
                    self.config["privacy"]["sigma"],
                )

            # Trust scoring
            trust_score = self.trust_manager.update_trust(
                client_id, update_vector, self.running_global_estimate
            )

            # Buffer (used only for running_global_estimate / diagnostics now)
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

            # Update running estimate with the processed vector
            if self.running_global_estimate is None:
                self.running_global_estimate = update_vector
            else:
                self.running_global_estimate = (
                    0.9 * self.running_global_estimate + 0.1 * update_vector
                )

        # Write the DP'd vector back so the caller (route) can pass it
        # to GroupManager for aggregation.
        client_update["local_updates"] = update_vector.tobytes()
        return client_update

    def _decode_update(self, encoded_update: Any) -> np.ndarray:
        """Decode client update from transport format."""
        if encoded_update is None:
            self.logger.warning("Received null update, returning zero array")
            return np.array([], dtype=np.float32)

        if isinstance(encoded_update, bytes):
            return np.frombuffer(encoded_update, dtype="<f4")
        elif isinstance(encoded_update, np.ndarray):
            return encoded_update
        else:
            return np.array(encoded_update, dtype=np.float32)
