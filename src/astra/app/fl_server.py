"""
Federated Learning Server orchestration.

Wraps the core engine's AsyncServer with the connection manager
and group manager, providing experiment lifecycle methods.
"""

import base64
import json
import logging
import time
from typing import Any

import numpy as np

from astra.app.database import get_db
from astra.app.group_manager import GroupManager
from astra.core.aggregation.aggregator import create_aggregator
from astra.core.data_splitter import DataSplitter
from astra.core.server import AsyncServer
from astra.core.utils.seed import set_seed
from astra.infra.connection_manager import ConnectionManager
from astra.infra.models import ClientUpdate
from astra.infra.registry import get_registry


class FLServer:
    """Federated Learning Server with API."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.connection_manager = ConnectionManager()
        self.db = get_db()
        self.group_manager = GroupManager(config, self.connection_manager)

        self.server: AsyncServer | None = None
        self.model_registry = get_registry()

        self.experiment_id: str | None = None
        self.is_running = False
        self.is_paused = False

        self.logger = logging.getLogger(__name__)

        self._setup_server()

    def _reload_models_from_db(self):
        """Reload external model registrations from the database."""
        import importlib
        import logging

        from astra.app.database import get_db

        logger = logging.getLogger(__name__)
        try:
            rows = get_db().load_model_registrations()
        except Exception:
            rows = []

        for row in rows:
            model_id = row.get("model_id")
            arch_path = row.get("architecture_path")
            source = row.get("source", "external")
            if not model_id or not arch_path:
                continue
            try:
                config_row = row.get("config_json")
                config_data = json.loads(config_row) if config_row else {}

                if source == "huggingface":
                    self.model_registry.register_hf_model(
                        model_name=arch_path,
                        use_peft=config_data.get("use_peft", False),
                        peft_config={
                            "enabled": config_data.get("use_peft", False),
                            "method": config_data.get("peft_method", "lora"),
                            "lora_rank": 8,
                            "lora_alpha": 16,
                            "target_modules": ["q_proj", "v_proj"],
                        } if config_data.get("use_peft") else {"enabled": False},
                    )
                else:
                    module_path, attr_name = arch_path.rsplit(".", 1)
                    module = importlib.import_module(module_path)
                    factory_fn = getattr(module, attr_name)
                    kwargs = config_data.get("kwargs", {})
                    model_info = {
                        "model_id": model_id,
                        "source": "external",
                        "architecture_path": arch_path,
                        "config": kwargs,
                    }
                    self.model_registry.register_factory(
                        model_id,
                        lambda fn=factory_fn, kw=kwargs: fn(**kw) if kw else fn(),
                        model_info,
                    )
                logger.info("Reloaded model '%s' from DB (source: %s)", model_id, source)
            except Exception as e:
                logger.warning("Failed to reload model '%s': %s", model_id, e)

    def _setup_server(self):
        """Initialize the FL server components."""
        # Reload externally registered models from DB
        self._reload_models_from_db()

        is_peft = self.config.get("peft", {}).get("enabled", False)

        if is_peft:
            from astra.core.models.hf_models import (
                load_hf_peft_model,
            )

            hf_config = self.config.get("model", {}).get("hf", {})
            model_name = hf_config.get(
                "hf_model_name", "openai/clip-vit-base-patch32"
            )
            peft_config = self.config.get("peft", {})
            model, _ = load_hf_peft_model(model_name, peft_config, device="cpu")
            model = model.to("cpu")

            model_id = f"hf_{model_name.replace('/', '_')}_peft"
            if model_id not in self.model_registry.model_instances:
                self.model_registry.model_instances[model_id] = model
        else:
            model_id = self.config.get("model", {}).get("model_id")
            if not model_id:
                self.logger.warning(
                    "No model_id in config — server starting without a global model. "
                    "Register a model via the dashboard (HuggingFace or External tabs) "
                    "and create a group with model_id."
                )
                model = None
            else:
                try:
                    model = self.model_registry.build_model(model_id)
                except ValueError:
                    raise RuntimeError(
                        f"Model '{model_id}' not found in registry. "
                        f"Available: {list(self.model_registry.model_factories.keys())}"
                    ) from None

        aggregator = create_aggregator(self.config)

        if model is not None:
            data_splitter = DataSplitter(self.config)
            _, val_loader = data_splitter.create_data_loaders()

            self.server = AsyncServer(
                model=model, aggregator=aggregator, config=self.config, val_loader=val_loader
            )
            self.group_manager.server_model = self.server.model
        else:
            self.server = None

        self.logger.info("FL Server initialized")

    async def handle_client_register(self, client_id: str, capabilities: dict) -> dict:
        """Handle client registration."""
        self.connection_manager.register_client(client_id, None)  # type: ignore[arg-type]
        self.db.register_fl_client(client_id, self.experiment_id or "default")

        self.logger.info(f"Client registered: {client_id}")

        return {"status": "registered", "client_id": client_id, "config": self.config}

    async def handle_client_update(self, update: ClientUpdate) -> dict:
        """Handle incoming client update."""
        if not self.server or not self.is_running or self.is_paused:
            return {"status": "rejected", "reason": "server_not_ready"}

        # Decode update
        try:
            delta_bytes = base64.b64decode(update.local_updates)
            delta = np.frombuffer(delta_bytes, dtype=np.float32)
        except Exception:
            delta = np.array([])

        client_update = {
            "client_id": update.client_id,
            "client_version": update.client_version,
            "local_updates": delta.tobytes(),
            "update_type": update.update_type,
            "local_dataset_size": update.local_dataset_size,
            "timestamp": time.time(),
            "meta": update.meta,
        }

        # Process update
        self.server.handle_update(client_update)

        # Broadcast update to dashboard
        await self.connection_manager.broadcast(
            {
                "type": "client_update",
                "client_id": update.client_id,
                "step": self.server.global_version,
            }
        )

        return {"status": "accepted", "global_version": self.server.global_version}

    async def get_global_model(self) -> dict:
        """Get global model state (simplified)."""
        if not self.server:
            return {}

        return {"global_version": self.server.global_version, "model_type": "custom"}

    def start_experiment(self, experiment_id: str, config: dict) -> None:
        """Start a new experiment."""
        self.experiment_id = experiment_id
        self.config = config

        set_seed(config.get("seed", 42))

        self.db.create_experiment(experiment_id, config)
        self.db.update_experiment_status(experiment_id, "running")

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
            self.db.update_experiment_status(self.experiment_id, "completed")
        self.logger.info("Experiment stopped")
