"""
Group Manager for federated learning.

Manages multiple training groups with hybrid async windowing,
client registration, aggregation, model saving, and training lifecycle.
"""

import asyncio
import contextlib
import json
import logging
import os
import threading
import time
import uuid
from datetime import datetime
from typing import Any

import numpy as np

from astra.app.database import get_db
from astra.app.training_group import AsyncWindowConfig, TrainingGroup
from astra.core.aggregation.aggregator import create_aggregator


class GroupManager:
    """Manages multiple training groups with hybrid async windowing."""

    def __init__(self, config: dict[str, Any], connection_manager=None):
        self.config = config
        self.groups: dict[str, TrainingGroup] = {}
        self.client_to_group: dict[str, str] = {}
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
        self.connection_manager = connection_manager
        self.training_tasks: dict[str, asyncio.Task] = {}
        self.server_model: Any = None

        self.event_logs: list[dict] = []

        self._load_groups_from_db()

        self._load_logs_from_db()

        self.logger.info("GroupManager init: %s groups loaded from DB", len(self.groups))

    # ------------------------------------------------------------------
    # Broadcasting
    # ------------------------------------------------------------------

    async def broadcast_to_group(self, group_id: str, message: dict):
        """Broadcast message to all clients in a group."""
        if not self.connection_manager:
            return
        group = self.groups.get(group_id)
        if not group:
            return
        for client_id in group.clients:
            await self.connection_manager.send_to(client_id, message)

    # ------------------------------------------------------------------
    # Database persistence
    # ------------------------------------------------------------------

    def _load_groups_from_db(self):
        """Load persisted groups from database on startup."""
        try:
            db = get_db()
            db_groups = db.get_all_groups()

            if db_groups:
                for g in db_groups:
                    gid = g["group_id"]
                    if gid in self.groups:
                        continue

                    config = (
                        json.loads(g.get("config_json", "{}"))
                        if isinstance(g.get("config_json"), str)
                        else (g.get("config_json") or {})
                    )
                    config.setdefault("auto_continue", False)

                    aggregator = create_aggregator(config)

                    group = TrainingGroup(
                        group_id=gid,
                        model_id=g.get("model_id", ""),
                        config=config,
                        join_token=g.get("join_token", ""),
                        window_config=AsyncWindowConfig(
                            window_size=g.get("window_size", 3),
                            time_limit=g.get("time_limit", 20.0),
                        ),
                        aggregator=aggregator,
                        max_rounds=config.get("max_rounds"),
                    )
                    group.status = g.get("status", "IDLE")

                    self.groups[gid] = group

                    # Reload FL clients for this group
                    try:
                        with db.connection() as conn:
                            client_rows = conn.execute(
                                "SELECT client_id, user_id,"
                                " trust_score, status, joined_at,"
                                " local_accuracy, local_loss,"
                                " updates_count, gradient_norm,"
                                " last_update"
                                " FROM fl_clients"
                                " WHERE group_id = ?",
                                (gid,),
                            ).fetchall()
                            for cr in client_rows:
                                cid = cr["client_id"]
                                group.clients[cid] = {
                                    "has_gpu": False,
                                    "device": "cpu",
                                    "data_metadata": {},
                                    "connection": "none",
                                    "last_update": cr["last_update"],
                                    "updates_count": cr["updates_count"] or 0,
                                    "local_accuracy": cr["local_accuracy"] or 0,
                                    "local_loss": cr["local_loss"] or 0,
                                    "trust_score": cr["trust_score"] or 1.0,
                                    "status": "offline",
                                    "joined_at": cr["joined_at"],
                                    "gradient_norm": cr["gradient_norm"] or 0,
                                }
                                self.client_to_group[cid] = gid
                    except Exception as e:
                        self.logger.warning(f"Could not load clients for group {gid}: {e}")

                    # Reload metrics history for this group
                    try:
                        with db.connection() as conn:
                            metric_rows = conn.execute(
                                "SELECT step, timestamp,"
                                " metrics_json FROM metrics"
                                " WHERE group_id = ?"
                                " ORDER BY step",
                                (gid,),
                            ).fetchall()
                            for mr in metric_rows:
                                m = json.loads(mr["metrics_json"])
                                group.metrics_history.append(m)
                            if metric_rows:
                                group.model_version = len(metric_rows)
                                group.completed_rounds = len(metric_rows)
                    except Exception as e:
                        self.logger.warning(f"Could not load metrics for group {gid}: {e}")

                    # Checkpoint resume: check if model file exists on disk
                    try:
                        latest_path = os.path.join("models", "global", gid, "model_latest.pt")
                        if os.path.exists(latest_path):
                            import torch

                            checkpoint = torch.load(
                                latest_path, map_location="cpu", weights_only=False
                            )
                            disk_version = checkpoint.get("version", 0)
                            if disk_version > group.model_version:
                                group.model_version = disk_version
                                group.completed_rounds = disk_version
                            self.logger.info(
                                f"Checkpoint found for {gid}:"
                                f" v{disk_version}"
                                f" (acc={checkpoint.get('accuracy', 0):.4f})"
                            )
                    except Exception as e:
                        self.logger.warning(f"Could not load checkpoint for group {gid}: {e}")

                    self.logger.info(
                        f"Restored group from DB: {gid}"
                        f" (status={group.status},"
                        f" clients={len(group.clients)},"
                        f" rounds={group.completed_rounds})"
                    )

                self.logger.info(f"Loaded {len(db_groups)} groups from database")
            else:
                self.logger.info("No groups found in database — create groups via the dashboard")
        except Exception as e:
            self.logger.warning(f"Could not load groups from DB: {e}")
            if not self.groups:
                self.logger.warning("Could not load groups from DB. Create via dashboard.")

    def _load_logs_from_db(self) -> None:
        """Load persisted event logs from database on startup."""
        try:
            db = get_db()
            # Load last 500 logs (same limit as in-memory ring buffer)
            db_logs = db.get_logs(limit=500)
            with self.lock:
                self.event_logs = db_logs
            self.logger.info(f"Loaded {len(self.event_logs)} event logs from database")
        except Exception as e:
            self.logger.warning(f"Could not load event logs from DB: {e}")

    # ------------------------------------------------------------------
    # Event logging
    # ------------------------------------------------------------------

    def log_event(
        self,
        event_type: str,
        message: str,
        group_id: str | None = None,
        details: dict | None = None,
    ):
        """Add an event to the log (in-memory + persisted to DB)."""
        timestamp = time.time()
        entry = {
            "timestamp": timestamp,
            "type": event_type,
            "message": message,
            "group_id": group_id,
            "details": details or {},
        }
        with self.lock:
            self.event_logs.append(entry)
            # Keep last 500 events in memory
            if len(self.event_logs) > 500:
                self.event_logs = self.event_logs[-500:]

        # Persist to database
        try:
            db = get_db()
            db.log_event(
                event_type=event_type,
                message=message,
                timestamp=timestamp,
                group_id=group_id,
                details=details,
            )
        except Exception as e:
            self.logger.warning(f"Could not persist event log to DB: {e}")

    def get_logs(
        self, limit: int = 100, event_type: str | None = None, group_id: str | None = None
    ) -> list[dict]:
        """Get recent logs from DB (primary) with in-memory fallback."""
        try:
            db = get_db()
            logs = db.get_logs(limit=limit, event_type=event_type, group_id=group_id)
            if logs:
                return logs
        except Exception as e:
            self.logger.warning(f"Could not read event logs from DB: {e}")

        # Fallback to in-memory list
        with self.lock:
            logs = list(self.event_logs)
            if event_type:
                logs = [e for e in logs if e["type"] == event_type]
            if group_id:
                logs = [e for e in logs if e.get("group_id") == group_id]
            return logs[-limit:][::-1]  # Most recent first

    # ------------------------------------------------------------------
    # Update decoding / normalisation
    # ------------------------------------------------------------------

    def _decode_local_updates(self, local_updates: Any) -> np.ndarray:
        """Decode base64/bytes/list updates into a float32 numpy array."""
        if local_updates is None:
            return np.array([], dtype=np.float32)
        if isinstance(local_updates, bytes):
            result = np.frombuffer(local_updates, dtype=np.float32)
            self.logger.debug("Decoded bytes update: %s elements", len(result))
            return result
        if isinstance(local_updates, str):
            try:
                import base64

                decoded = base64.b64decode(local_updates)
                result = np.frombuffer(decoded, dtype=np.float32)
                self.logger.debug("Decoded base64 update: %s elements", len(result))
                return result
            except Exception as e:
                self.logger.error("Failed to decode base64 update: %s", e, exc_info=True)
                return np.array([], dtype=np.float32)
        if isinstance(local_updates, np.ndarray):
            return local_updates.astype(np.float32)
        return np.array(local_updates, dtype=np.float32)

    def normalize_update(self, update: dict) -> dict:
        """Ensure updates have fields expected by aggregators."""
        if "delta" not in update:
            update["delta"] = self._decode_local_updates(update.get("local_updates"))
        delta = update.get("delta")
        if (
            delta is not None
            and hasattr(delta, "__len__")
            and len(delta) > 0
            and (np.any(np.isnan(delta)) or np.any(np.isinf(delta)))
        ):
            client_id = update.get("client_id", "unknown")
            self.logger.warning("Client %s: rejecting NaN/Inf update", client_id)
            update["delta"] = np.zeros_like(delta)
        update.setdefault("dataset_size", update.get("local_dataset_size", 1))
        update.setdefault("staleness_weight", 1.0)
        update.setdefault("trust", 1.0)
        return update

    # ------------------------------------------------------------------
    # Training watchdog (time-based aggregation)
    # ------------------------------------------------------------------

    def _start_training_watchdog(self, group_id: str) -> None:
        """Ensure a background task is running to enforce time-based aggregation."""
        task = self.training_tasks.get(group_id)
        if task and not task.done():
            return
        self.training_tasks[group_id] = asyncio.create_task(self._training_watchdog(group_id))

    def _stop_training_watchdog(self, group_id: str) -> None:
        task = self.training_tasks.pop(group_id, None)
        if task and not task.done():
            task.cancel()

    async def _training_watchdog(self, group_id: str) -> None:
        """Trigger aggregation on timeouts so training keeps progressing."""
        try:
            while True:
                await asyncio.sleep(1.0)
                with self.lock:
                    group = self.groups.get(group_id)
                    if not group or not group.is_training:
                        break
                    if not group.window_config.enabled:
                        continue
                    pending = len(group.pending_updates)
                    elapsed = time.time() - group.last_aggregation_time
                    time_limit = group.window_config.time_limit

                if pending == 0 or elapsed < time_limit:
                    continue

                agg_result = self.aggregate_group(group_id)
                if not agg_result:
                    continue

                await self.broadcast_to_group(
                    group_id,
                    {
                        "type": "model_update",
                        "version": agg_result["version"],
                        "group_id": group_id,
                        "accuracy": agg_result.get("accuracy", 0),
                        "loss": agg_result.get("loss", 0),
                    },
                )

                if group and group.is_training and group.config.get("auto_continue", False):
                    await self.trigger_clients_training(group_id)
        except asyncio.CancelledError:
            return

    # ------------------------------------------------------------------
    # Group CRUD
    # ------------------------------------------------------------------

    def create_group(
        self,
        group_id: str,
        model_id: str,
        config: dict[str, Any],
        window_size: int = 3,
        time_limit: float = 20.0,
    ) -> TrainingGroup:
        """Create a new training group."""
        with self.lock:
            if group_id in self.groups:
                return self.groups[group_id]

            config = config or {}
            config.setdefault("auto_continue", False)

            # Generate or use provided join token
            join_token = config.get("join_token")
            if not join_token or join_token == "GENERATE_NEW":
                join_token = uuid.uuid4().hex[:16]

            # Create aggregator for this group
            aggregator = create_aggregator(config)

            group = TrainingGroup(
                group_id=group_id,
                model_id=model_id,
                config=config,
                join_token=join_token,
                window_config=AsyncWindowConfig(window_size=window_size, time_limit=time_limit),
                aggregator=aggregator,
                max_rounds=config.get("max_rounds"),
            )

            self.groups[group_id] = group
            self.logger.info(f"Created group: {group_id}")

            # Persist to database
            try:
                db = get_db()
                db.create_group(
                    group_id=group_id,
                    model_id=model_id,
                    config=config,
                    join_token=join_token,
                    window_size=window_size,
                    time_limit=int(time_limit),
                )
            except Exception as e:
                self.logger.warning(f"Could not persist group {group_id} to DB: {e}")

            # If PEFT is enabled, save the base model to disk for client downloads
            self._save_hf_model_to_disk(group_id, model_id, config)

            return group

    def _save_hf_model_to_disk(
        self, group_id: str, model_id: str, config: dict[str, Any]
    ) -> None:
        """Save HuggingFace base model to disk so clients can download it.

        Called after group creation when PEFT is enabled. Saves both .pt and
        safetensors formats to ``models/hf/{model_id}/``.
        """
        is_peft = config.get("peft", {}).get("enabled", False)

        # Also check the model registry — PEFT info lives there when
        # the group was created via the HuggingFace tab
        if not is_peft:
            try:
                from astra.infra.registry import get_registry as _get_reg

                _reg = _get_reg()
                _info = _reg.get_model_info(model_id)
                if _info:
                    is_peft = _info.get("is_peft", False)
            except Exception:
                pass

        if not is_peft:
            return

        try:
            from astra.infra.registry import get_registry

            registry = get_registry()
            if model_id not in registry.model_instances:
                self.logger.debug(
                    "No model instance for '%s' — deferring disk save to first aggregation",
                    model_id,
                )
                return

            model = registry.model_instances[model_id]
            peft_config = config.get("peft", {})

            save_dir = os.path.join("models", "hf", model_id)
            from astra.core.models.hf_models import save_base_model_to_disk

            saved = save_base_model_to_disk(
                model=model,
                save_dir=save_dir,
                model_name=model_id,
                peft_config=peft_config,
            )
            self.log_event(
                "hf_model_saved",
                f"Saved HuggingFace base model for group {group_id} to disk",
                group_id,
                {"files": {k: os.path.basename(v) for k, v in saved.items()}, "save_dir": save_dir},
            )
            self.logger.info(
                "Saved HF base model for group %s -> %s (%s)",
                group_id,
                save_dir,
                list(saved.keys()),
            )
        except Exception as e:
            self.logger.warning(
                "Could not save HF model to disk for group %s: %s", group_id, e
            )

    def delete_group(self, group_id: str) -> bool:
        with self.lock:
            if group_id not in self.groups:
                return False

            for client_id, g_id in list(self.client_to_group.items()):
                if g_id == group_id:
                    del self.client_to_group[client_id]

            del self.groups[group_id]

            # Remove from database
            try:
                db = get_db()
                db.delete_group(group_id)
            except Exception as e:
                self.logger.warning(f"Could not delete group {group_id} from DB: {e}")

            return True

    # ------------------------------------------------------------------
    # Client management
    # ------------------------------------------------------------------

    def register_client(
        self,
        client_id: str,
        group_id: str,
        client_info: dict | None = None,
    ) -> bool:
        """Register client to a group."""
        with self.lock:
            if group_id not in self.groups:
                return False

            # Check if already in another group
            if client_id in self.client_to_group:
                current = self.client_to_group[client_id]
                if current != group_id:
                    self.log_event(
                        "client_rejected",
                        f"Client {client_id} tried to migrate from {current} to {group_id}",
                        group_id,
                    )
                    return False  # No migration allowed

            group = self.groups[group_id]

            # Auto-start training when first client joins
            if len(group.clients) == 0 and not group.is_training:
                group.is_locked = True
                group.is_training = True
                group.status = "TRAINING"
                group.completed_rounds = 0
                self._start_training_watchdog(group_id)
                self.log_event(
                    "training_started",
                    f"Training auto-started for group {group_id} (first client joined)",
                    group_id,
                )

            group.add_client(client_id, client_info)
            self.client_to_group[client_id] = group_id

            # Persist FL client to database
            try:
                db = get_db()
                user_id = client_info.get("user_id") if client_info else None
                db.register_fl_client(
                    client_id=client_id, experiment_id=group_id, user_id=user_id, group_id=group_id
                )
            except Exception as e:
                self.logger.warning(f"Could not persist client {client_id} to DB: {e}")

            self.log_event(
                "client_joined",
                f"Client {client_id} joined group {group_id}",
                group_id,
                {"client_id": client_id},
            )

            return True

    def get_client_group(self, client_id: str) -> TrainingGroup | None:
        group_id = self.client_to_group.get(client_id)
        return self.groups.get(group_id) if group_id else None

    # ------------------------------------------------------------------
    # Updates & aggregation
    # ------------------------------------------------------------------

    def _is_peft_group(self, group: TrainingGroup) -> bool:
        """Check if a group uses PEFT — from group config or model registry."""
        if group.config.get("peft", {}).get("enabled", False):
            return True
        try:
            from astra.infra.registry import get_registry as _get_reg

            _reg = _get_reg()
            _info = _reg.get_model_info(group.model_id)
            if _info:
                return _info.get("is_peft", False)
        except Exception:
            pass
        return False

    def add_client_update(self, client_id: str, update: dict) -> dict | None:
        """Add update and check if aggregation triggered (hybrid windowing)."""
        with self.lock:
            group = self.get_client_group(client_id)
            if not group:
                return None

            normalized = self.normalize_update(update)
            triggered = group.add_update(client_id, normalized)

            result = {
                "group_id": group.group_id,
                "triggered": triggered,
                "window_status": group.get_window_status(),
            }

            if triggered:
                result["aggregate"] = True

            return result

    def aggregate_group(self, group_id: str) -> dict | None:
        """Aggregate updates in a group's buffer."""
        with self.lock:
            if group_id not in self.groups:
                self.logger.warning("aggregate_group: unknown group %s", group_id)
                return None

            group = self.groups[group_id]

            if len(group.pending_updates) == 0:
                self.logger.debug("aggregate_group: %s has no pending updates", group_id)
                return None

            self.logger.info(
                "aggregate_group: %s aggregating %s updates (round %s)",
                group_id,
                len(group.pending_updates),
                group.completed_rounds + 1,
            )

            updates = [self.normalize_update(u["update"]) for u in group.pending_updates]
            client_ids = [u["client_id"] for u in group.pending_updates]

            accuracies = [u.get("meta", {}).get("train_accuracy", 0) for u in updates]
            losses = [u.get("meta", {}).get("train_loss", 0) for u in updates]

            global_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
            global_loss = sum(losses) / len(losses) if losses else 0

            if group.aggregator:
                aggregated = group.aggregator.aggregate(updates)
            else:
                aggregated = np.mean([u.get("delta", np.array([])) for u in updates], axis=0)

            self.logger.info(
                "aggregate_group: %s result — acc=%.4f, loss=%.4f, clients=%s",
                group_id,
                global_accuracy,
                global_loss,
                client_ids,
            )

            # Apply aggregated delta to the live server model
            if self.server_model is not None and len(aggregated) > 0:
                is_peft = self._is_peft_group(group)
                if is_peft:
                    from astra.core.models.model_zoo import apply_peft_delta as _apply
                else:
                    from astra.core.models.model_zoo import apply_flat_delta as _apply
                _apply(self.server_model, aggregated)

            # Update version
            group.model_version += 1
            group.completed_rounds += 1

            # Store metrics
            group.metrics_history.append(
                {
                    "version": group.model_version,
                    "timestamp": time.time(),
                    "accuracy": global_accuracy,
                    "loss": global_loss,
                    "clients": len(updates),
                }
            )

            # Persist metrics to database
            try:
                db = get_db()
                db.log_metrics(
                    experiment_id=self.experiment_id or "default",
                    step=group.model_version,
                    metrics={
                        "version": group.model_version,
                        "timestamp": time.time(),
                        "accuracy": global_accuracy,
                        "loss": global_loss,
                        "clients": len(updates),
                    },
                    group_id=group_id,
                )
            except Exception as e:
                self.logger.warning(f"Could not persist metrics for group {group_id}: {e}")

            group.clear_updates()

            self.log_event(
                "aggregation",
                f"Aggregated {len(updates)} updates -> v{group.model_version}",
                group_id,
                {
                    "version": group.model_version,
                    "clients": len(updates),
                    "accuracy": global_accuracy,
                    "loss": global_loss,
                },
            )

            # Save global model weights to disk
            self.save_model_weights(
                group_id=group_id,
                model_version=group.model_version,
                aggregated_weights=aggregated,
                accuracy=global_accuracy,
                loss=global_loss,
                num_clients=len(updates),
            )

            self.logger.info(
                f"Aggregated group {group_id}:"
                f" {len(updates)} clients,"
                f" v{group.model_version},"
                f" acc={global_accuracy:.4f},"
                f" loss={global_loss:.4f}"
            )

            # Broadcast to all connected WebSocket clients (including dashboard)
            if self.connection_manager:
                with contextlib.suppress(RuntimeError):
                    asyncio.create_task(
                        self.connection_manager.broadcast(
                            {
                                "type": "aggregation_complete",
                                "group_id": group_id,
                                "version": group.model_version,
                                "accuracy": global_accuracy,
                                "loss": global_loss,
                                "contributing_clients": len(updates),
                                "completed_rounds": group.completed_rounds,
                                "timestamp": time.time(),
                            }
                        )
                    )

            if group.max_rounds is not None and group.completed_rounds >= group.max_rounds:
                group.is_training = False
                group.status = "COMPLETED"
                self._stop_training_watchdog(group_id)
                self.log_event(
                    "training_completed",
                    f"Training completed for group {group_id}",
                    group_id,
                    {"version": group.model_version, "rounds": group.completed_rounds},
                )

            return {
                "group_id": group_id,
                "version": group.model_version,
                "accuracy": global_accuracy,
                "loss": global_loss,
                "contributing_clients": client_ids,
                "update_count": len(updates),
                "aggregated_model": aggregated,
            }

    # ------------------------------------------------------------------
    # Model persistence
    # ------------------------------------------------------------------

    def save_model_weights(
        self,
        group_id: str,
        model_version: int,
        aggregated_weights,
        accuracy: float,
        loss: float,
        num_clients: int,
    ):
        """Save global model weights and adapter checkpoints to disk."""
        try:
            import torch

            save_dir = os.path.join("models", "global", group_id)
            os.makedirs(save_dir, exist_ok=True)

            file_path = os.path.join(save_dir, f"model_v{model_version}.pt")
            torch.save(
                {
                    "version": model_version,
                    "weights": aggregated_weights,
                    "accuracy": accuracy,
                    "loss": loss,
                    "num_clients": num_clients,
                    "timestamp": datetime.now().isoformat(),
                    "group_id": group_id,
                },
                file_path,
            )

            latest_path = os.path.join(save_dir, "model_latest.pt")
            torch.save(
                {
                    "version": model_version,
                    "weights": aggregated_weights,
                    "accuracy": accuracy,
                    "loss": loss,
                    "num_clients": num_clients,
                    "timestamp": datetime.now().isoformat(),
                    "group_id": group_id,
                },
                latest_path,
            )

            group_obj = self.groups.get(group_id)
            if self.server_model is not None and group_obj and self._is_peft_group(group_obj):
                from astra.core.models.hf_models import (
                    get_base_model_state_dict,
                    get_lora_state_dict,
                    save_base_model_to_disk,
                )

                base_path = os.path.join(save_dir, "base.pt")
                if not os.path.exists(base_path):
                    base_state = get_base_model_state_dict(self.server_model)
                    torch.save({"base_state_dict": base_state}, base_path)
                    self.logger.info(f"Saved base model → {base_path}")

                # Also save to models/hf/{model_id}/ for client downloads (if not already there)
                hf_save_dir = os.path.join("models", "hf", group_obj.model_id)
                if not os.path.exists(os.path.join(hf_save_dir, "base_model.pt")):
                    try:
                        save_base_model_to_disk(
                            model=self.server_model,
                            save_dir=hf_save_dir,
                            model_name=group_obj.model_id,
                            peft_config=group_obj.config.get("peft", self.config.get("peft", {})),
                        )
                    except Exception as e:
                        self.logger.debug("Could not save HF base model to disk: %s", e)

                adapter_state = get_lora_state_dict(self.server_model)
                adapter_path = os.path.join(save_dir, f"adapter_v{model_version}.pt")
                torch.save({"lora_state_dict": adapter_state}, adapter_path)

                adapter_latest = os.path.join(save_dir, "adapter_latest.pt")
                torch.save({"lora_state_dict": adapter_state}, adapter_latest)
                self.logger.info(
                    f"Saved adapter v{model_version} ({len(adapter_state)} params)"
                )

            db = get_db()
            db.save_model_record(
                group_id=group_id,
                model_type="global",
                file_path=file_path,
                version=model_version,
                accuracy=accuracy,
                loss=loss,
                num_clients=num_clients,
            )

            self.logger.info(
                f"Saved global model v{model_version} for group {group_id} → {file_path}"
            )
        except Exception as e:
            self.logger.warning(f"Could not save model for group {group_id}: {e}")

    # ------------------------------------------------------------------
    # Group queries
    # ------------------------------------------------------------------

    def get_all_groups(self, include_secret: bool = False) -> list[dict]:
        with self.lock:
            return [g.to_dict(include_secret) for g in self.groups.values()]

    # ------------------------------------------------------------------
    # Training lifecycle
    # ------------------------------------------------------------------

    def start_group_training(self, group_id: str) -> bool:
        """Start training for a group."""
        with self.lock:
            if group_id not in self.groups:
                return False
            group = self.groups[group_id]
            if group.is_locked:
                return False

            group.is_locked = True
            group.is_training = True
            group.status = "TRAINING"
            group.completed_rounds = 0

            self._start_training_watchdog(group_id)

            self.log_event("training_started", f"Group {group_id} ready to accept deltas", group_id)
            self.logger.info(f"Started training for group {group_id}")
            return True

    async def notify_training_started(self, group_id: str):
        """Broadcast a model_update to all clients so they can pull the new global model.

        Clients train externally and call POST /api/clients/{id}/delta to submit updates.
        This broadcast tells the dashboard that a new aggregated version is available.
        """
        group = self.groups.get(group_id)
        if not group:
            return

        self.log_event(
            "aggregation_broadcast",
            f"Aggregation complete for group {group_id}, new global model available",
            group_id,
            {"client_count": len(group.clients), "version": group.model_version},
        )

        await self.broadcast_to_group(
            group_id,
            {
                "type": "model_update",
                "group_id": group_id,
                "version": group.model_version,
            },
        )

    def process_client_update(self, client_id: str, update: dict) -> dict:
        """Process client update and check if aggregation needed."""
        group = self.get_client_group(client_id)
        if not group:
            return {"triggered": False, "group_id": None}

        normalized = self.normalize_update(update)
        triggered = group.add_update(client_id, normalized)

        # Persist client metrics to database
        try:
            db = get_db()
            client_info = group.clients.get(client_id, {})
            db.update_fl_client_metrics(
                client_id=client_id,
                local_accuracy=client_info.get("local_accuracy", 0),
                local_loss=client_info.get("local_loss", 0),
                updates_count=client_info.get("updates_count", 0),
                gradient_norm=client_info.get("gradient_norm", 0),
                status="active",
            )
        except Exception as e:
            self.logger.warning(f"Could not persist metrics for client {client_id}: {e}")

        result = {
            "triggered": triggered,
            "group_id": group.group_id,
            "window_status": group.get_window_status(),
        }

        if triggered:
            result["aggregate"] = True

        return result

    def pause_group_training(self, group_id: str) -> bool:
        """Pause training for a group."""
        with self.lock:
            if group_id not in self.groups:
                return False
            group = self.groups[group_id]
            group.is_training = False
            group.status = "PAUSED"
            self._stop_training_watchdog(group_id)
            return True

    def resume_group_training(self, group_id: str) -> bool:
        """Resume training for a group."""
        with self.lock:
            if group_id not in self.groups:
                return False
            group = self.groups[group_id]
            group.is_training = True
            group.status = "TRAINING"
            self._start_training_watchdog(group_id)
            return True

    def stop_group_training(self, group_id: str) -> bool:
        """Stop training for a group."""
        with self.lock:
            if group_id not in self.groups:
                return False
            group = self.groups[group_id]
            group.is_training = False
            group.status = "COMPLETED"
            self._stop_training_watchdog(group_id)
            return True

    def get_all_client_status(self) -> list[dict]:
        clients = []
        for group_id, group in self.groups.items():
            for client_id, info in group.clients.items():
                clients.append({"client_id": client_id, "group_id": group_id, **info})
        return clients
