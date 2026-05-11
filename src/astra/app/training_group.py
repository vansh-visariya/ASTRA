"""
Training group data structures for federated learning.

Contains AsyncWindowConfig and TrainingGroup dataclasses that represent
the core state of a federated learning training group.
"""

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


# ============================================================================
# Hybrid Async Window Configuration
# ============================================================================

@dataclass
class AsyncWindowConfig:
    """Configuration for hybrid async windowing."""
    window_size: int = 3  # Aggregate when N updates received
    time_limit: float = 20.0  # OR after T seconds elapsed
    enabled: bool = True


# ============================================================================
# Training Group
# ============================================================================

@dataclass
class TrainingGroup:
    """Represents a federated learning training group."""
    group_id: str
    model_id: str
    config: Dict[str, Any]
    window_config: AsyncWindowConfig = field(default_factory=AsyncWindowConfig)

    # Security - join token (secret)
    join_token: str = field(default_factory=lambda: uuid.uuid4().hex[:16])

    # Model state
    model_version: int = 0
    model: Any = None

    # Update buffer for hybrid windowing
    pending_updates: List[Dict] = field(default_factory=list)
    last_aggregation_time: float = field(default_factory=time.time)

    # Client tracking
    clients: Dict[str, Dict] = field(default_factory=dict)

    # Aggregator
    aggregator: Any = None

    # Status
    status: str = 'CREATED'  # CREATED, WAITING, READY, TRAINING, PAUSED, COMPLETED, FAILED
    is_training: bool = False
    is_locked: bool = False  # Lock config when training starts
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # Training bounds
    max_rounds: Optional[int] = None
    completed_rounds: int = 0

    # Metrics
    metrics_history: List[Dict] = field(default_factory=list)

    def add_client(self, client_id: str, client_info: Dict = None) -> None:
        if client_id not in self.clients:
            self.clients[client_id] = {
                'status': 'active',
                'joined_at': datetime.now().isoformat(),
                'last_update': None,
                'trust_score': 1.0,
                'updates_count': 0,
                'local_accuracy': 0.0,
                'local_loss': 0.0,
                'gradient_norm': 0.0,
                **(client_info or {})
            }

    def remove_client(self, client_id: str) -> None:
        if client_id in self.clients:
            self.clients[client_id]['status'] = 'disconnected'

    def add_update(self, client_id: str, update: Dict) -> bool:
        """Add client update to buffer. Returns True if aggregation triggered."""
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"[ADD-UPDATE] Client {client_id} in group {self.group_id}. Group clients: {list(self.clients.keys())}")
        if client_id not in self.clients:
            logger.warning(f"⚠️  Client {client_id} NOT in group {self.group_id} clients. Available: {list(self.clients.keys())}")
        if client_id in self.clients:
            self.clients[client_id]['last_update'] = time.time()
            self.clients[client_id]['updates_count'] += 1
            train_acc = update.get('meta', {}).get('train_accuracy', 0)
            train_loss = update.get('meta', {}).get('train_loss', 0)
            self.clients[client_id]['local_accuracy'] = train_acc
            self.clients[client_id]['local_loss'] = train_loss
            self.clients[client_id]['gradient_norm'] = update.get('meta', {}).get('gradient_norm', 0)
            logger.info(f"✓ METRICS STORED: Client {client_id} in group {self.group_id} | acc={train_acc:.4f}, loss={train_loss:.4f}")

        self.pending_updates.append({
            'client_id': client_id,
            'update': update,
            'timestamp': time.time()
        })

        # Check hybrid triggers
        size_triggered = len(self.pending_updates) >= self.window_config.window_size
        time_triggered = (time.time() - self.last_aggregation_time) >= self.window_config.time_limit

        return size_triggered or time_triggered

    def get_window_status(self) -> Dict:
        """Get current window status for UI."""
        elapsed = time.time() - self.last_aggregation_time
        return {
            'pending_updates': len(self.pending_updates),
            'window_size': self.window_config.window_size,
            'time_elapsed': round(elapsed, 1),
            'time_limit': self.window_config.time_limit,
            'time_remaining': max(0, self.window_config.time_limit - elapsed),
            'size_triggered': len(self.pending_updates) >= self.window_config.window_size,
            'time_triggered': elapsed >= self.window_config.time_limit,
            'trigger_reason': 'size' if len(self.pending_updates) >= self.window_config.window_size else ('time' if elapsed >= self.window_config.time_limit else 'waiting')
        }

    def clear_updates(self) -> None:
        """Clear buffer after aggregation."""
        self.pending_updates.clear()
        self.last_aggregation_time = time.time()

    def get_active_clients(self) -> List[str]:
        return [cid for cid, info in self.clients.items() if info.get('status') == 'active']

    def to_dict(self, include_secret: bool = False) -> Dict:
        return {
            'group_id': self.group_id,
            'model_id': self.model_id,
            'model_version': self.model_version,
            'status': self.status,
            'is_training': self.is_training,
            'is_locked': self.is_locked,
            'created_at': self.created_at,
            'completed_rounds': self.completed_rounds,
            'max_rounds': self.max_rounds,
            'join_token': self.join_token if include_secret else '***HIDDEN***',
            'config': {
                'local_epochs': self.config.get('local_epochs', 2),
                'batch_size': self.config.get('batch_size', 32),
                'lr': self.config.get('lr', 0.01),
                'aggregator': self.config.get('aggregator', 'fedavg'),
                'dp_enabled': self.config.get('dp_enabled', False),
            },
            'window_config': {
                'window_size': self.window_config.window_size,
                'time_limit': self.window_config.time_limit,
                'enabled': self.window_config.enabled
            },
            'window_status': self.get_window_status(),
            'client_count': len(self.clients),
            'active_clients': self.get_active_clients(),
            'pending_updates': len(self.pending_updates),
            'metrics_history': self.metrics_history[-10:]  # Last 10 entries
        }
