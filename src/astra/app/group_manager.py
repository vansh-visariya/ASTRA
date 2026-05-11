"""
Group Manager for federated learning.

Manages multiple training groups with hybrid async windowing,
client registration, aggregation, model saving, and training lifecycle.
"""

import asyncio
import json
import logging
import os
import threading
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from api.database import get_db
from core_engine.aggregator import create_aggregator

from networking.training_group import AsyncWindowConfig, TrainingGroup


class GroupManager:
    """Manages multiple training groups with hybrid async windowing."""

    def __init__(self, config: Dict[str, Any], connection_manager=None):
        self.config = config
        self.groups: Dict[str, TrainingGroup] = {}
        self.client_to_group: Dict[str, str] = {}
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
        self.connection_manager = connection_manager
        self.training_tasks: Dict[str, asyncio.Task] = {}

        # Event logs
        self.event_logs: List[Dict] = []

        self._load_groups_from_db()

    # ------------------------------------------------------------------
    # Broadcasting
    # ------------------------------------------------------------------

    async def broadcast_to_group(self, group_id: str, message: Dict):
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
                    gid = g['group_id']
                    if gid in self.groups:
                        continue

                    config = json.loads(g.get('config_json', '{}')) if isinstance(g.get('config_json'), str) else (g.get('config_json') or {})
                    config.setdefault('auto_continue', False)

                    aggregator = create_aggregator(config)

                    group = TrainingGroup(
                        group_id=gid,
                        model_id=g.get('model_id', 'simple_cnn_mnist'),
                        config=config,
                        join_token=g.get('join_token', ''),
                        window_config=AsyncWindowConfig(
                            window_size=g.get('window_size', 3),
                            time_limit=g.get('time_limit', 20.0)
                        ),
                        aggregator=aggregator,
                        max_rounds=config.get('max_rounds')
                    )
                    group.status = g.get('status', 'IDLE')

                    self.groups[gid] = group

                    # Reload FL clients for this group
                    try:
                        with db.connection() as conn:
                            client_rows = conn.execute(
                                "SELECT client_id, user_id, trust_score, status, joined_at FROM fl_clients WHERE group_id = ?",
                                (gid,)
                            ).fetchall()
                            for cr in client_rows:
                                cid = cr['client_id']
                                group.clients[cid] = {
                                    'has_gpu': False,
                                    'device': 'cpu',
                                    'data_metadata': {},
                                    'connection': 'none',
                                    'last_update': None,
                                    'updates_count': 0,
                                    'local_accuracy': 0,
                                    'local_loss': 0,
                                    'trust_score': cr['trust_score'] or 1.0,
                                    'status': 'offline',
                                    'joined_at': cr['joined_at'],
                                    'gradient_norm': 0,
                                }
                                self.client_to_group[cid] = gid
                    except Exception as e:
                        self.logger.warning(f"Could not load clients for group {gid}: {e}")

                    # Reload metrics history for this group
                    try:
                        with db.connection() as conn:
                            metric_rows = conn.execute(
                                "SELECT step, timestamp, metrics_json FROM metrics WHERE group_id = ? ORDER BY step",
                                (gid,)
                            ).fetchall()
                            for mr in metric_rows:
                                m = json.loads(mr['metrics_json'])
                                group.metrics_history.append(m)
                            if metric_rows:
                                group.model_version = len(metric_rows)
                                group.completed_rounds = len(metric_rows)
                    except Exception as e:
                        self.logger.warning(f"Could not load metrics for group {gid}: {e}")

                    # Checkpoint resume: check if model file exists on disk
                    try:
                        latest_path = os.path.join('models', 'global', gid, 'model_latest.pt')
                        if os.path.exists(latest_path):
                            import torch
                            checkpoint = torch.load(latest_path, map_location='cpu', weights_only=False)
                            disk_version = checkpoint.get('version', 0)
                            if disk_version > group.model_version:
                                group.model_version = disk_version
                                group.completed_rounds = disk_version
                            self.logger.info(f"Checkpoint found for {gid}: v{disk_version} (acc={checkpoint.get('accuracy', 0):.4f})")
                    except Exception as e:
                        self.logger.warning(f"Could not load checkpoint for group {gid}: {e}")

                    self.logger.info(f"Restored group from DB: {gid} (status={group.status}, clients={len(group.clients)}, rounds={group.completed_rounds})")

                self.logger.info(f"Loaded {len(db_groups)} groups from database")
            else:
                # No persisted groups - create default
                self.create_group(
                    group_id='default',
                    model_id='simple_cnn_mnist',
                    config={},
                    window_size=3,
                    time_limit=20.0
                )
                self.logger.info("Created default group (no groups in DB)")
        except Exception as e:
            self.logger.warning(f"Could not load groups from DB: {e}")
            if not self.groups:
                aggregator = create_aggregator({})
                group = TrainingGroup(
                    group_id='default',
                    model_id='simple_cnn_mnist',
                    config={},
                    join_token=uuid.uuid4().hex[:16],
                    window_config=AsyncWindowConfig(window_size=3, time_limit=20.0),
                    aggregator=aggregator
                )
                self.groups['default'] = group

    # ------------------------------------------------------------------
    # Event logging
    # ------------------------------------------------------------------

    def log_event(self, event_type: str, message: str, group_id: str = None, details: Dict = None):
        """Add an event to the log."""
        with self.lock:
            self.event_logs.append({
                'timestamp': time.time(),
                'type': event_type,
                'message': message,
                'group_id': group_id,
                'details': details or {}
            })
            # Keep last 500 events
            if len(self.event_logs) > 500:
                self.event_logs = self.event_logs[-500:]

    def get_logs(self, limit: int = 100, event_type: str = None, group_id: str = None) -> List[Dict]:
        """Get recent logs."""
        with self.lock:
            logs = list(self.event_logs)
            if event_type:
                logs = [l for l in logs if l['type'] == event_type]
            if group_id:
                logs = [l for l in logs if l.get('group_id') == group_id]
            return logs[-limit:][::-1]  # Most recent first

    # ------------------------------------------------------------------
    # Update decoding / normalisation
    # ------------------------------------------------------------------

    def _decode_local_updates(self, local_updates: Any) -> np.ndarray:
        """Decode base64/bytes/list updates into a float32 numpy array."""
        if local_updates is None:
            return np.array([], dtype=np.float32)
        if isinstance(local_updates, bytes):
            return np.frombuffer(local_updates, dtype=np.float32)
        if isinstance(local_updates, str):
            try:
                import base64
                decoded = base64.b64decode(local_updates)
                return np.frombuffer(decoded, dtype=np.float32)
            except Exception:
                return np.array([], dtype=np.float32)
        if isinstance(local_updates, np.ndarray):
            return local_updates.astype(np.float32)
        return np.array(local_updates, dtype=np.float32)

    def normalize_update(self, update: Dict) -> Dict:
        """Ensure updates have fields expected by aggregators."""
        logger = logging.getLogger(__name__)
        logger.debug(f"NORMALIZE: Input meta={update.get('meta', {}).get('train_accuracy', 0)}")
        if 'delta' not in update:
            update['delta'] = self._decode_local_updates(update.get('local_updates'))
        # Validate: reject NaN/Inf updates that would poison the global model
        delta = update.get('delta')
        if delta is not None and hasattr(delta, '__len__') and len(delta) > 0:
            if np.any(np.isnan(delta)) or np.any(np.isinf(delta)):
                logger.warning("Rejecting update with NaN/Inf values")
                update['delta'] = np.zeros_like(delta)
        update.setdefault('dataset_size', update.get('local_dataset_size', 1))
        update.setdefault('staleness_weight', 1.0)
        update.setdefault('trust', 1.0)
        logger.debug(f"NORMALIZE: Output meta={update.get('meta', {}).get('train_accuracy', 0)}")
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

                await self.broadcast_to_group(group_id, {
                    'type': 'model_update',
                    'version': agg_result['version'],
                    'group_id': group_id,
                    'accuracy': agg_result.get('accuracy', 0),
                    'loss': agg_result.get('loss', 0)
                })

                if group and group.is_training and group.config.get('auto_continue', False):
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
        config: Dict[str, Any],
        window_size: int = 3,
        time_limit: float = 20.0
    ) -> TrainingGroup:
        """Create a new training group."""
        with self.lock:
            if group_id in self.groups:
                return self.groups[group_id]

            config = config or {}
            config.setdefault('auto_continue', False)

            # Generate or use provided join token
            join_token = config.get('join_token')
            if not join_token or join_token == "GENERATE_NEW":
                join_token = uuid.uuid4().hex[:16]

            # Create aggregator for this group
            aggregator = create_aggregator(config)

            group = TrainingGroup(
                group_id=group_id,
                model_id=model_id,
                config=config,
                join_token=join_token,
                window_config=AsyncWindowConfig(
                    window_size=window_size,
                    time_limit=time_limit
                ),
                aggregator=aggregator,
                max_rounds=config.get('max_rounds')
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
                    time_limit=int(time_limit)
                )
            except Exception as e:
                self.logger.warning(f"Could not persist group {group_id} to DB: {e}")

            return group

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

    def register_client(self, client_id: str, group_id: str, client_info: Dict = None) -> bool:
        """Register client to a group."""
        with self.lock:
            if group_id not in self.groups:
                return False

            # Check if already in another group
            if client_id in self.client_to_group:
                current = self.client_to_group[client_id]
                if current != group_id:
                    self.log_event('client_rejected', f'Client {client_id} tried to migrate from {current} to {group_id}', group_id)
                    return False  # No migration allowed

            group = self.groups[group_id]

            # Auto-start training when first client joins
            if len(group.clients) == 0 and not group.is_training:
                group.is_locked = True
                group.is_training = True
                group.status = 'TRAINING'
                group.completed_rounds = 0
                self._start_training_watchdog(group_id)
                self.log_event('training_started', f'Training auto-started for group {group_id} (first client joined)', group_id)

            group.add_client(client_id, client_info)
            self.client_to_group[client_id] = group_id

            # Persist FL client to database
            try:
                db = get_db()
                user_id = client_info.get('user_id') if client_info else None
                db.register_fl_client(
                    client_id=client_id,
                    experiment_id=group_id,
                    user_id=user_id,
                    group_id=group_id
                )
            except Exception as e:
                self.logger.warning(f"Could not persist client {client_id} to DB: {e}")

            self.log_event('client_joined', f'Client {client_id} joined group {group_id}', group_id, {'client_id': client_id})

            return True

    def get_client_group(self, client_id: str) -> Optional[TrainingGroup]:
        group_id = self.client_to_group.get(client_id)
        return self.groups.get(group_id) if group_id else None

    # ------------------------------------------------------------------
    # Updates & aggregation
    # ------------------------------------------------------------------

    def add_client_update(self, client_id: str, update: Dict) -> Optional[Dict]:
        """Add update and check if aggregation triggered (hybrid windowing)."""
        with self.lock:
            group = self.get_client_group(client_id)
            if not group:
                return None

            normalized = self.normalize_update(update)
            triggered = group.add_update(client_id, normalized)

            result = {
                'group_id': group.group_id,
                'triggered': triggered,
                'window_status': group.get_window_status()
            }

            if triggered:
                result['aggregate'] = True

            return result

    def aggregate_group(self, group_id: str) -> Optional[Dict]:
        """Aggregate updates in a group's buffer."""
        with self.lock:
            if group_id not in self.groups:
                return None

            group = self.groups[group_id]

            if len(group.pending_updates) == 0:
                return None

            # Get all updates
            updates = [self.normalize_update(u['update']) for u in group.pending_updates]
            client_ids = [u['client_id'] for u in group.pending_updates]

            # Calculate global metrics from client updates
            accuracies = [u.get('meta', {}).get('train_accuracy', 0) for u in updates]
            losses = [u.get('meta', {}).get('train_loss', 0) for u in updates]

            global_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
            global_loss = sum(losses) / len(losses) if losses else 0

            # Aggregate model weights
            if group.aggregator:
                aggregated = group.aggregator.aggregate(updates)
            else:
                aggregated = np.mean([u.get('delta', np.array([])) for u in updates], axis=0)

            # Update version
            group.model_version += 1
            group.completed_rounds += 1

            # Store metrics
            group.metrics_history.append({
                'version': group.model_version,
                'timestamp': time.time(),
                'accuracy': global_accuracy,
                'loss': global_loss,
                'clients': len(updates)
            })

            # Persist metrics to database
            try:
                db = get_db()
                db.log_metrics(
                    experiment_id=group_id,
                    step=group.model_version,
                    metrics={
                        'version': group.model_version,
                        'timestamp': time.time(),
                        'accuracy': global_accuracy,
                        'loss': global_loss,
                        'clients': len(updates)
                    },
                    group_id=group_id
                )
            except Exception as e:
                self.logger.warning(f"Could not persist metrics for group {group_id}: {e}")

            group.clear_updates()

            self.log_event('aggregation', f'Aggregated {len(updates)} updates -> v{group.model_version}', group_id, {
                'version': group.model_version,
                'clients': len(updates),
                'accuracy': global_accuracy,
                'loss': global_loss
            })

            # Save global model weights to disk
            self.save_model_weights(
                group_id=group_id,
                model_version=group.model_version,
                aggregated_weights=aggregated,
                accuracy=global_accuracy,
                loss=global_loss,
                num_clients=len(updates)
            )

            self.logger.info(
                f"Aggregated group {group_id}: {len(updates)} clients, v{group.model_version}, acc={global_accuracy:.4f}, loss={global_loss:.4f}"
            )

            # Broadcast to all connected WebSocket clients (including dashboard)
            if self.connection_manager:
                try:
                    asyncio.create_task(self.connection_manager.broadcast({
                        'type': 'aggregation_complete',
                        'group_id': group_id,
                        'version': group.model_version,
                        'accuracy': global_accuracy,
                        'loss': global_loss,
                        'contributing_clients': len(updates),
                        'completed_rounds': group.completed_rounds,
                        'timestamp': time.time()
                    }))
                except RuntimeError:
                    pass  # No event loop running (e.g., during tests)

            if group.max_rounds is not None and group.completed_rounds >= group.max_rounds:
                group.is_training = False
                group.status = 'COMPLETED'
                self._stop_training_watchdog(group_id)
                self.log_event('training_completed', f'Training completed for group {group_id}', group_id, {
                    'version': group.model_version,
                    'rounds': group.completed_rounds
                })

            return {
                'group_id': group_id,
                'version': group.model_version,
                'accuracy': global_accuracy,
                'loss': global_loss,
                'contributing_clients': client_ids,
                'update_count': len(updates),
                'aggregated_model': aggregated
            }

    # ------------------------------------------------------------------
    # Model persistence
    # ------------------------------------------------------------------

    def save_model_weights(self, group_id: str, model_version: int,
                           aggregated_weights, accuracy: float, loss: float,
                           num_clients: int):
        """Save global model weights to disk and record in DB."""
        try:
            import torch
            save_dir = os.path.join('models', 'global', group_id)
            os.makedirs(save_dir, exist_ok=True)

            file_path = os.path.join(save_dir, f'model_v{model_version}.pt')
            torch.save({
                'version': model_version,
                'weights': aggregated_weights,
                'accuracy': accuracy,
                'loss': loss,
                'num_clients': num_clients,
                'timestamp': datetime.now().isoformat(),
                'group_id': group_id
            }, file_path)

            # Also save as latest
            latest_path = os.path.join(save_dir, 'model_latest.pt')
            torch.save({
                'version': model_version,
                'weights': aggregated_weights,
                'accuracy': accuracy,
                'loss': loss,
                'num_clients': num_clients,
                'timestamp': datetime.now().isoformat(),
                'group_id': group_id
            }, latest_path)

            # Record in database
            db = get_db()
            db.save_model_record(
                group_id=group_id,
                model_type='global',
                file_path=file_path,
                version=model_version,
                accuracy=accuracy,
                loss=loss,
                num_clients=num_clients
            )

            self.logger.info(f"Saved global model v{model_version} for group {group_id} → {file_path}")
        except Exception as e:
            self.logger.warning(f"Could not save model for group {group_id}: {e}")

    # ------------------------------------------------------------------
    # Group queries
    # ------------------------------------------------------------------

    def get_all_groups(self, include_secret: bool = False) -> List[Dict]:
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
            group.status = 'TRAINING'
            group.completed_rounds = 0

            self._start_training_watchdog(group_id)

            self.log_event('training_started', f'Training started for group {group_id}', group_id)
            self.logger.info(f"Started training for group {group_id}")
            return True

    async def notify_training_started(self, group_id: str):
        """Notify all clients that training is open - they should begin autonomous training."""
        group = self.groups.get(group_id)
        if not group:
            return

        self.log_event('training_started_notify', f'Training opened for group {group_id}, clients may begin', group_id, {
            'client_count': len(group.clients)
        })

        await self.broadcast_to_group(group_id, {
            'type': 'training_started',
            'group_id': group_id,
            'config': {
                'local_epochs': group.config.get('local_epochs', 2),
                'batch_size': group.config.get('batch_size', 32),
                'lr': group.config.get('lr', 0.01),
            }
        })

    async def trigger_clients_training(self, group_id: str):
        """Explicitly trigger a new local training round for all clients in a group."""
        group = self.groups.get(group_id)
        if not group:
            return

        await self.broadcast_to_group(group_id, {
            'type': 'train_command',
            'group_id': group_id,
            'config': {
                'local_epochs': group.config.get('local_epochs', 2),
                'batch_size': group.config.get('batch_size', 32),
                'lr': group.config.get('lr', 0.01),
            }
        })

    async def notify_training_paused(self, group_id: str):
        """Notify all clients that training is paused."""
        await self.broadcast_to_group(group_id, {
            'type': 'training_paused',
            'group_id': group_id,
        })

    async def notify_training_stopped(self, group_id: str):
        """Notify all clients that training is stopped."""
        await self.broadcast_to_group(group_id, {
            'type': 'training_stopped',
            'group_id': group_id,
        })

    def process_client_update(self, client_id: str, update: Dict) -> Dict:
        """Process client update and check if aggregation needed."""
        group = self.get_client_group(client_id)
        if not group:
            return {'triggered': False, 'group_id': None}

        normalized = self.normalize_update(update)
        triggered = group.add_update(client_id, normalized)

        result = {
            'triggered': triggered,
            'group_id': group.group_id,
            'window_status': group.get_window_status()
        }

        if triggered:
            result['aggregate'] = True

        return result

    def pause_group_training(self, group_id: str) -> bool:
        """Pause training for a group."""
        with self.lock:
            if group_id not in self.groups:
                return False
            group = self.groups[group_id]
            group.is_training = False
            group.status = 'PAUSED'
            self._stop_training_watchdog(group_id)
            return True

    def resume_group_training(self, group_id: str) -> bool:
        """Resume training for a group."""
        with self.lock:
            if group_id not in self.groups:
                return False
            group = self.groups[group_id]
            group.is_training = True
            group.status = 'TRAINING'
            self._start_training_watchdog(group_id)
            return True

    def stop_group_training(self, group_id: str) -> bool:
        """Stop training for a group."""
        with self.lock:
            if group_id not in self.groups:
                return False
            group = self.groups[group_id]
            group.is_training = False
            group.status = 'COMPLETED'
            self._stop_training_watchdog(group_id)
            return True

    def get_all_client_status(self) -> List[Dict]:
        clients = []
        for group_id, group in self.groups.items():
            for client_id, info in group.clients.items():
                clients.append({
                    'client_id': client_id,
                    'group_id': group_id,
                    **info
                })
        return clients
