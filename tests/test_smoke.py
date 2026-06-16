"""
Smoke tests for the ASTRA server API.

Verifies that the server starts and basic endpoints respond.
"""


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
    assert resp.status_code == 200
    data = resp.json()
    assert "running" in data
    assert "connected_clients" in data


def test_system_metrics(client):
    resp = client.get("/api/system/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_groups" in data
    assert "active_groups" in data
    assert "total_participants" in data


def test_groups_list(client):
    resp = client.get("/api/groups")
    assert resp.status_code == 401


def test_models_list(client):
    resp = client.get("/api/models")
    assert resp.status_code == 200
    data = resp.json()
    assert "models" in data
    assert "count" in data


def test_clients_list(client):
    resp = client.get("/api/clients")
    assert resp.status_code == 200
    data = resp.json()
    assert "clients" in data
    assert "count" in data
