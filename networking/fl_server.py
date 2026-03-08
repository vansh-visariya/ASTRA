"""
Federated Learning Server orchestration.

Wraps the core engine's AsyncServer with the connection manager
and group manager, providing experiment lifecycle methods.
"""

import base64
import logging
import time
from typing import Any, Dict, Optional

import numpy as np

from api.database import get_db
from core_engine.aggregator import create_aggregator
from core_engine.data_splitter import DataSplitter
from core_engine.server import AsyncServer
from core_engine.utils.seed import set_seed
from model_registry.registry import get_registry

from networking.connection_manager import ConnectionManager
from networking.group_manager import GroupManager
from networking.models import ClientUpdate


class FLServer:
    """Federated Learning Server with API."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.connection_manager = ConnectionManager()
        self.db = get_db()
        self.group_manager = GroupManager(config, self.connection_manager)

        self.server: Optional[AsyncServer] = None
        self.model_registry = get_registry()

        self.experiment_id: Optional[str] = None
        self.is_running = False
        self.is_paused = False

        self.logger = logging.getLogger(__name__)

        self._setup_server()

    def _setup_server(self):
        """Initialize the FL server."""
        from core_engine.model_zoo import create_model

        # Create model
        model = create_model(self.config)

        # Create aggregator
        aggregator = create_aggregator(self.config)

        # Create data splitter (for validation)
        data_splitter = DataSplitter(self.config)
        _, val_loader = data_splitter.create_data_loaders()

        # Create async server
        self.server = AsyncServer(
            model=model,
            aggregator=aggregator,
            config=self.config,
            val_loader=val_loader
        )

        self.logger.info("FL Server initialized")

    async def handle_client_register(self, client_id: str, capabilities: Dict) -> Dict:
        """Handle client registration."""
        self.connection_manager.register_client(client_id, None)
        self.db.register_fl_client(client_id, self.experiment_id or 'default')

        self.logger.info(f"Client registered: {client_id}")

        return {
            'status': 'registered',
            'client_id': client_id,
            'config': self.config
        }

    async def handle_client_update(self, update: ClientUpdate) -> Dict:
        """Handle incoming client update."""
        if not self.server or not self.is_running or self.is_paused:
            return {'status': 'rejected', 'reason': 'server_not_ready'}

        # Decode update
        try:
            delta_bytes = base64.b64decode(update.local_updates)
            delta = np.frombuffer(delta_bytes, dtype=np.float32)
        except Exception:
            delta = np.array([])

        client_update = {
            'client_id': update.client_id,
            'client_version': update.client_version,
            'local_updates': delta.tobytes(),
            'update_type': update.update_type,
            'local_dataset_size': update.local_dataset_size,
            'timestamp': time.time(),
            'meta': update.meta
        }

        # Process update
        self.server.handle_update(client_update)

        # Broadcast update to dashboard
        await self.connection_manager.broadcast({
            'type': 'client_update',
            'client_id': update.client_id,
            'step': self.server.global_version
        })

        return {'status': 'accepted', 'global_version': self.server.global_version}

    async def get_global_model(self) -> Dict:
        """Get global model state (simplified)."""
        if not self.server:
            return {}

        return {
            'global_version': self.server.global_version,
            'model_type': 'simple_cnn'
        }

    def start_experiment(self, experiment_id: str, config: Dict) -> None:
        """Start a new experiment."""
        self.experiment_id = experiment_id
        self.config = config

        set_seed(config.get('seed', 42))

        self.db.create_experiment(experiment_id, config)
        self.db.update_experiment_status(experiment_id, 'running')

        self.is_running = True
        self.is_paused = False

        self._setup_server()

        self.logger.info(f"Experiment started: {experiment_id}")

    def pause_experiment(self) -> None:
        self.is_paused = True
        self.logger.info("Experiment paused")

    def resume_experiment(self) -> None:
        self.is_paused = False
        self.logger.info("Experiment resumed")

    def stop_experiment(self) -> None:
        self.is_running = False
        if self.experiment_id:
            self.db.update_experiment_status(self.experiment_id, 'completed')
        self.logger.info("Experiment stopped")
