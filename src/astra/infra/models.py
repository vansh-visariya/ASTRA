"""
Pydantic request/response models for the Federated Learning API.
"""

from typing import Any

from pydantic import BaseModel


class ClientRegister(BaseModel):
    client_id: str
    capabilities: dict[str, Any] = {}


class ClientUpdate(BaseModel):
    client_id: str
    client_version: int
    local_updates: str  # Base64 encoded
    update_type: str = "delta"
    local_dataset_size: int
    meta: dict[str, Any] = {}


class ExperimentConfig(BaseModel):
    experiment_id: str
    config: dict[str, Any]


class ControlCommand(BaseModel):
    command: str  # start, pause, resume, stop
    params: dict[str, Any] = {}
