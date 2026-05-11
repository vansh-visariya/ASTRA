"""
Pydantic request/response models for the Federated Learning API.
"""

from typing import Any, Dict
from pydantic import BaseModel


class ClientRegister(BaseModel):
    client_id: str
    capabilities: Dict[str, Any] = {}


class ClientUpdate(BaseModel):
    client_id: str
    client_version: int
    local_updates: str  # Base64 encoded
    update_type: str = "delta"
    local_dataset_size: int
    meta: Dict[str, Any] = {}


class ExperimentConfig(BaseModel):
    experiment_id: str
    config: Dict[str, Any]


class ControlCommand(BaseModel):
    command: str  # start, pause, resume, stop
    params: Dict[str, Any] = {}
