"""
Group Management System for Federated Learning.

Handles:
- Group creation and management
- Client membership tracking
- Per-group aggregation and model state
- Hybrid async windowing (size + time based)
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from collections import defaultdict
import threading

import numpy as np


@dataclass
class TrainingGroup:
    """Represents a federated learning group."""
    
    group_id: str
    model_id: str
    config: Dict[str, Any]
    
    # Model state
    model_version: int = 0
    model: Any = None
    
    # Async window config
    window_size: int = 3
    time_limit: float = 20.0  # seconds
    
    # Update buffer
    pending_updates: List[Dict] = field(default_factory=list)
    last_aggregation_time: float = field(default_factory=time.time)
    
    # Client tracking
    clients: Dict[str, Dict] = field(default_factory=dict)
    
    # Aggregator
    aggregator: Any = None
    
    # Status
    is_training: bool = False
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Metrics
    metrics_history: List[Dict] = field(default_factory=list)
    
    def add_client(self, client_id: str, client_info: Dict = None) -> None:
        """Add client to group."""
        if client_id not in self.clients:
            self.clients[client_id] = {
                'status': 'active',
                'joined_at': datetime.now().isoformat(),
                'last_update': None,
                'trust_score': 1.0,
                'updates_count': 0,
                **(client_info or {})
            }
    
    def remove_client(self, client_id: str) -> None:
        """Remove client from group."""
        if client_id in self.clients:
            self.clients[client_id]['status'] = 'disconnected'
    
    def add_update(self, client_id: str, update: Dict) -> bool:
        """Add client update to buffer. Returns True if window triggered."""
        if client_id in self.clients:
            self.clients[client_id]['last_update'] = time.time()
            self.clients[client_id]['updates_count'] += 1
        
        self.pending_updates.append({
            'client_id': client_id,
            'update': update,
            'timestamp': time.time()
        })
        
        # Check triggers
        size_triggered = len(self.pending_updates) >= self.window_size
        time_triggered = (time.time() - self.last_aggregation_time) >= self.time_limit
        
        return size_triggered or time_triggered
    
    def get_window_status(self) -> Dict:
        """Get current window status."""
        elapsed = time.time() - self.last_aggregation_time
        return {
            'pending_updates': len(self.pending_updates),
            'window_size': self.window_size,
            'time_elapsed': round(elapsed, 1),
            'time_limit': self.time_limit,
            'time_remaining': max(0, self.time_limit - elapsed),
            'size_triggered': len(self.pending_updates) >= self.window_size,
            'time_triggered': elapsed >= self.time_limit
        }
    
    def clear_updates(self) -> None:
        """Clear update buffer after aggregation."""
        self.pending_updates.clear()
        self.last_aggregation_time = time.time()
    
    def get_active_clients(self) -> List[str]:
        """Get list of active client IDs."""
        return [cid for cid, info in self.clients.items() 
                if info.get('status') == 'active']
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'group_id': self.group_id,
            'model_id': self.model_id,
            'model_version': self.model_version,
            'is_training': self.is_training,
            'created_at': self.created_at,
            'clients': self.clients,
            'window_status': self.get_window_status(),
            'metrics_count': len(self.metrics_history)
        }


class GroupManager:
    """
    Manages multiple training groups.
    
    Features:
    - Create/delete groups
    - Client registration
    - Hybrid async windowing
    - Per-group aggregation
    - Model versioning
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.groups: Dict[str, TrainingGroup] = {}
        self.client_to_group: Dict[str, str] = {}  # client_id -> group_id
        
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
        
        self._init_default_groups()
    
    def _init_default_groups(self) -> None:
        """Initialize default groups from config."""
        default_groups = self.config.get('groups', [
            {
                'group_id': 'default',
                'model_id': 'simple_cnn_mnist',
                'window_size': 3,
                'time_limit': 20.0
            }
        ])
        
        for g in default_groups:
            self.create_group(
                group_id=g['group_id'],
                model_id=g.get('model_id', 'simple_cnn_mnist'),
                config=g.get('config', {}),
                window_size=g.get('window_size', 3),
                time_limit=g.get('time_limit', 20.0)
            )
    
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
                self.logger.warning(f"Group {group_id} already exists")
                return self.groups[group_id]
            
            group = TrainingGroup(
                group_id=group_id,
                model_id=model_id,
                config=config,
                window_size=window_size,
                time_limit=time_limit
            )
            
            self.groups[group_id] = group
            self.logger.info(f"Created group: {group_id}")
            
            return group
    
    def delete_group(self, group_id: str) -> bool:
        """Delete a group."""
        with self.lock:
            if group_id not in self.groups:
                return False
            
            # Remove client mappings
            for client_id, g_id in list(self.client_to_group.items()):
                if g_id == group_id:
                    del self.client_to_group[client_id]
            
            del self.groups[group_id]
            self.logger.info(f"Deleted group: {group_id}")
            return True
    
    def register_client(
        self,
        client_id: str,
        group_id: str,
        client_info: Dict = None
    ) -> bool:
        """Register client to a group."""
        with self.lock:
            if group_id not in self.groups:
                self.logger.error(f"Group {group_id} does not exist")
                return False
            
            # Check if client already in another group
            if client_id in self.client_to_group:
                current_group = self.client_to_group[client_id]
                if current_group != group_id:
                    self.logger.warning(
                        f"Client {client_id} already in group {current_group}"
                    )
                    return False
            
            group = self.groups[group_id]
            group.add_client(client_id, client_info)
            self.client_to_group[client_id] = group_id
            
            self.logger.info(f"Client {client_id} registered to group {group_id}")
            return True
    
    def unregister_client(self, client_id: str) -> bool:
        """Unregister client from group."""
        with self.lock:
            if client_id not in self.client_to_group:
                return False
            
            group_id = self.client_to_group[client_id]
            group = self.groups[group_id]
            group.remove_client(client_id)
            
            del self.client_to_group[client_id]
            
            self.logger.info(f"Client {client_id} unregistered from {group_id}")
            return True
    
    def get_client_group(self, client_id: str) -> Optional[TrainingGroup]:
        """Get client's group."""
        group_id = self.client_to_group.get(client_id)
        if group_id:
            return self.groups.get(group_id)
        return None
    
    def add_client_update(
        self,
        client_id: str,
        update: Dict
    ) -> Optional[Dict]:
        """Add client update and check if aggregation triggered."""
        with self.lock:
            group = self.get_client_group(client_id)
            if not group:
                self.logger.error(f"Client {client_id} not in any group")
                return None
            
            triggered = group.add_update(client_id, update)
            
            if triggered:
                return {
                    'group_id': group.group_id,
                    'triggered': True,
                    'pending_count': len(group.pending_updates),
                    'window_status': group.get_window_status()
                }
            
            return {
                'group_id': group.group_id,
                'triggered': False,
                'pending_count': len(group.pending_updates),
                'window_status': group.get_window_status()
            }
    
    def aggregate_group(self, group_id: str) -> Optional[Dict]:
        """Manually trigger aggregation for a group."""
        with self.lock:
            if group_id not in self.groups:
                return None
            
            group = self.groups[group_id]
            
            if len(group.pending_updates) == 0:
                return None
            
            # Get all updates
            updates = [u['update'] for u in group.pending_updates]
            client_ids = [u['client_id'] for u in group.pending_updates]
            
            # Aggregate
            if group.aggregator:
                aggregated = group.aggregator.aggregate(updates)
            else:
                # Simple average fallback
                aggregated = np.mean([np.array(u) for u in updates], axis=0)
            
            # Update version
            group.model_version += 1
            
            # Clear buffer
            group.clear_updates()
            
            self.logger.info(
                f"Aggregated group {group_id}: "
                f"{len(updates)} updates, version {group.model_version}"
            )
            
            return {
                'group_id': group_id,
                'version': group.model_version,
                'contributing_clients': client_ids,
                'update_count': len(updates),
                'aggregated_model': aggregated
            }
    
    def get_all_groups(self) -> List[Dict]:
        """Get all groups as dictionaries."""
        with self.lock:
            return [g.to_dict() for g in self.groups.values()]
    
    def get_group_status(self, group_id: str) -> Optional[Dict]:
        """Get status of a specific group."""
        group = self.groups.get(group_id)
        if not group:
            return None
        
        return {
            **group.to_dict(),
            'window_status': group.get_window_status(),
            'active_clients': group.get_active_clients()
        }
    
    def get_all_client_status(self) -> List[Dict]:
        """Get status of all clients across groups."""
        clients = []
        
        for group_id, group in self.groups.items():
            for client_id, info in group.clients.items():
                clients.append({
                    'client_id': client_id,
                    'group_id': group_id,
                    **info
                })
        
        return clients
