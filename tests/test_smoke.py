"""
Smoke tests for the ASTRA server API.

Verifies that the server starts and basic endpoints respond.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock


@pytest.fixture
def client():
    from astra.app import state
    from astra.app.server_api import app
    from astra.infra.registry import get_registry

    registry = MagicMock()
    registry.list_models.return_value = []
    registry.model_factories = {}

    mock_server = MagicMock()
    mock_server.server = MagicMock()
    mock_server.server.running = False
    mock_server.is_running = False
    mock_server.group_manager = MagicMock()
    mock_server.group_manager.groups = {}
    mock_server.group_manager.event_logs = []
    mock_server.group_manager.get_all_groups.return_value = []
    mock_server.model_registry = registry
    mock_server.connection_manager = MagicMock()
    mock_server.connection_manager.get_connected_clients.return_value = []

    state.set_fl_server(mock_server)

    client = TestClient(app)
    return client


def test_root(client):
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["message"] == "Federated Learning API"
    assert data["version"] == "1.0.0"


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"


def test_server_status(client):
    resp = client.get("/api/server/status")
    assert resp.status_code == 401  # requires auth now


def test_system_metrics(client):
    resp = client.get("/api/system/metrics")
    assert resp.status_code == 401  # requires auth now


def test_groups_list(client):
    resp = client.get("/api/groups")
    assert resp.status_code == 401


def test_models_list(client):
    resp = client.get("/api/models")
    assert resp.status_code == 401  # requires auth now


def test_clients_list(client):
    resp = client.get("/api/clients")
    assert resp.status_code == 401  # requires auth now
