"""
Networking package for the ASTRA Federated Learning platform.

Key modules:
- server_api: FastAPI application entry point
- fl_server: FLServer orchestration class
- group_manager: GroupManager for training group lifecycle
- training_group: TrainingGroup and AsyncWindowConfig dataclasses
- connection_manager: WebSocket connection management
- models: Pydantic request/response models
- state: Shared server state accessor
- websocket_handler: WebSocket and Socket.IO handlers
- routes/: REST API endpoint modules
"""

from networking.models import ClientRegister, ClientUpdate, ExperimentConfig, ControlCommand
from networking.training_group import AsyncWindowConfig, TrainingGroup
from networking.connection_manager import ConnectionManager
from networking.group_manager import GroupManager
from networking.fl_server import FLServer
from networking.state import get_fl_server
