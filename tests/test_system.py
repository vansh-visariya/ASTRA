"""
System tests: Full REST API endpoint coverage with authentication,
group lifecycle, model registration, join requests, and edge cases.
"""

import base64
import json
import os
import tempfile

import pytest
from fastapi.testclient import TestClient

from astra.app.server_api import app


@pytest.fixture
def fresh_client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def auth_headers(fresh_client):
    resp = fresh_client.post("/api/auth/signup", json={
        "username": f"test_admin_{os.urandom(4).hex()}",
        "password": "testpass123",
        "role": "admin",
    })
    data = resp.json()
    login = fresh_client.post("/api/auth/login", json={
        "username": data.get("user", {}).get("username", ""),
        "password": "testpass123",
    })
    if login.status_code != 200:
        login = fresh_client.post("/api/auth/login", json={
            "username": f"test_admin_{os.urandom(4).hex()}",
            "password": "testpass123",
        })
        if login.status_code != 200:
            login = fresh_client.post("/api/auth/login", json={
                "username": "admin",
                "password": os.environ.get("ASTRA_DEFAULT_ADMIN_PASSWORD", "admin"),
            })

    token = login.json().get("token", "")
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def client_headers(fresh_client):
    username = f"test_client_{os.urandom(4).hex()}"
    fresh_client.post("/api/auth/signup", json={
        "username": username,
        "password": "testpass123",
        "role": "client",
    })
    resp = fresh_client.post("/api/auth/login", json={
        "username": username,
        "password": "testpass123",
    })
    token = resp.json().get("token", "")
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def group_id(fresh_client, auth_headers):
    resp = fresh_client.post("/api/groups", json={
        "group_id": f"test_grp_{os.urandom(4).hex()}",
        "model_id": "",
        "window_size": 3,
        "time_limit": 20,
    }, headers=auth_headers)
    if resp.status_code == 200:
        return resp.json()["group"]["group_id"]
    return f"test_grp_{os.urandom(4).hex()}"


class TestAuthEndpoints:
    def test_signup(self, fresh_client):
        resp = fresh_client.post("/api/auth/signup", json={
            "username": f"new_user_{os.urandom(4).hex()}",
            "password": "securepass123",
            "role": "client",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "user" in data
        assert data["user"]["role"] == "client"

    def test_signup_duplicate(self, fresh_client):
        username = f"dup_{os.urandom(4).hex()}"
        fresh_client.post("/api/auth/signup", json={
            "username": username, "password": "pass123", "role": "client",
        })
        resp = fresh_client.post("/api/auth/signup", json={
            "username": username, "password": "pass456", "role": "client",
        })
        assert resp.status_code == 400

    def test_login(self, fresh_client):
        username = f"login_{os.urandom(4).hex()}"
        fresh_client.post("/api/auth/signup", json={
            "username": username, "password": "pass123", "role": "client",
        })
        resp = fresh_client.post("/api/auth/login", json={
            "username": username, "password": "pass123",
        })
        assert resp.status_code == 200
        assert "token" in resp.json()

    def test_login_wrong_password(self, fresh_client):
        username = f"wrongp_{os.urandom(4).hex()}"
        fresh_client.post("/api/auth/signup", json={
            "username": username, "password": "correct", "role": "client",
        })
        resp = fresh_client.post("/api/auth/login", json={
            "username": username, "password": "wrong",
        })
        assert resp.status_code != 200


class TestGroupEndpoints:
    def test_create_group_auth_required(self, fresh_client):
        resp = fresh_client.post("/api/groups", json={
            "group_id": "no_auth",
            "model_id": "",
        })
        assert resp.status_code == 401

    def test_create_and_get_group(self, fresh_client, auth_headers):
        mid = f"info_{os.urandom(4).hex()}"
        fresh_client.post("/api/models/register/architecture", json={
            "model_id": mid,
            "architecture_path": "torch.nn.Linear",
            "config": {"in_features": 10, "out_features": 3},
        }, headers=auth_headers)
        gid = f"g_create_{os.urandom(4).hex()}"
        resp = fresh_client.post("/api/groups", json={
            "group_id": gid,
            "model_id": mid,
        }, headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["group"]["group_id"] == gid

        resp2 = fresh_client.get(f"/api/groups/{gid}", headers=auth_headers)
        assert resp2.status_code == 200
        assert resp2.json()["group"]["group_id"] == gid

    def test_list_groups_requires_auth(self, fresh_client):
        resp = fresh_client.get("/api/groups")
        assert resp.status_code == 401

    def test_list_groups_with_auth(self, fresh_client, auth_headers):
        resp = fresh_client.get("/api/groups", headers=auth_headers)
        assert resp.status_code == 200
        assert "groups" in resp.json()

    def test_delete_group(self, fresh_client, auth_headers):
        mid = f"del_m_{os.urandom(4).hex()}"
        fresh_client.post("/api/models/register/architecture", json={
            "model_id": mid,
            "architecture_path": "torch.nn.Linear",
            "config": {"in_features": 10, "out_features": 3},
        }, headers=auth_headers)
        gid = f"g_del_{os.urandom(4).hex()}"
        fresh_client.post("/api/groups", json={
            "group_id": gid, "model_id": mid,
        }, headers=auth_headers)

        resp = fresh_client.delete(f"/api/groups/{gid}", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"

        resp2 = fresh_client.get(f"/api/groups/{gid}", headers=auth_headers)
        assert resp2.status_code == 404

    def test_get_nonexistent_group(self, fresh_client, auth_headers):
        resp = fresh_client.get("/api/groups/does_not_exist_xyz", headers=auth_headers)
        assert resp.status_code == 404


class TestModelEndpoints:
    def test_list_models(self, fresh_client, auth_headers):
        resp = fresh_client.get("/api/models", headers=auth_headers)
        assert resp.status_code == 200
        assert "models" in resp.json()

    def test_register_architecture(self, fresh_client, auth_headers):
        resp = fresh_client.post("/api/models/register/architecture", json={
            "model_id": f"arch_{os.urandom(4).hex()}",
            "architecture_path": "torch.nn.Linear",
            "config": {"in_features": 10, "out_features": 5},
        }, headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["status"] == "registered"

    def test_register_duplicate(self, fresh_client, auth_headers):
        mid = f"dup_arch_{os.urandom(4).hex()}"
        fresh_client.post("/api/models/register/architecture", json={
            "model_id": mid,
            "architecture_path": "torch.nn.Linear",
            "config": {"in_features": 10, "out_features": 3},
        }, headers=auth_headers)

        resp = fresh_client.post("/api/models/register/architecture", json={
            "model_id": mid,
            "architecture_path": "torch.nn.Linear",
            "config": {"in_features": 5, "out_features": 2},
        }, headers=auth_headers)
        assert resp.status_code == 400

    def test_get_model_info(self, fresh_client, auth_headers):
        resp = fresh_client.post("/api/models/register/architecture", json={
            "model_id": f"info_{os.urandom(4).hex()}",
            "architecture_path": "torch.nn.Linear",
            "config": {"in_features": 10, "out_features": 3},
        }, headers=auth_headers)
        mid = resp.json()["model"]["model_id"]
        resp2 = fresh_client.get(f"/api/models/{mid}", headers=auth_headers)
        assert resp2.status_code == 200


class TestJoinRequestEndpoints:
    def test_client_join_request(self, fresh_client, client_headers, auth_headers):
        mid = f"jr2_{os.urandom(4).hex()}"
        fresh_client.post("/api/models/register/architecture", json={
            "model_id": mid,
            "architecture_path": "torch.nn.Linear",
            "config": {"in_features": 10, "out_features": 3},
        }, headers=auth_headers)
        gid = f"j_{os.urandom(4).hex()}"
        fresh_client.post("/api/groups", json={
            "group_id": gid, "model_id": mid,
        }, headers=auth_headers)

        resp = fresh_client.post("/api/join/join-request", json={
            "group_id": gid,
        }, headers=client_headers)
        assert resp.status_code == 200

    def test_duplicate_join_request(self, fresh_client, client_headers, auth_headers):
        gid = f"jd_{os.urandom(4).hex()}"
        fresh_client.post("/api/groups", json={
            "group_id": gid, "model_id": "",
        }, headers=auth_headers)

        fresh_client.post("/api/join/join-request", json={
            "group_id": gid,
        }, headers=client_headers)

        resp = fresh_client.post("/api/join/join-request", json={
            "group_id": gid,
        }, headers=client_headers)
        assert resp.status_code == 400

    def test_join_nonexistent_group(self, fresh_client, client_headers):
        resp = fresh_client.post("/api/join/join-request", json={
            "group_id": "nonexistent_group_99999",
        }, headers=client_headers)
        assert resp.status_code == 400

    def test_pending_requests_admin_only(self, fresh_client, client_headers, auth_headers):
        resp = fresh_client.get("/api/join/join-requests", headers=client_headers)
        assert resp.status_code == 403

        resp2 = fresh_client.get("/api/join/join-requests", headers=auth_headers)
        assert resp2.status_code == 200


class TestClientEndpoints:
    def test_list_clients(self, fresh_client, auth_headers):
        resp = fresh_client.get("/api/clients", headers=auth_headers)
        assert resp.status_code == 200

    def test_register_client(self, fresh_client, auth_headers):
        gid = f"reg_{os.urandom(4).hex()}"
        fresh_client.post("/api/groups", json={
            "group_id": gid, "model_id": "",
        }, headers=auth_headers)

        resp = fresh_client.post("/api/clients/register", json={
            "client_id": f"c_{os.urandom(4).hex()}",
            "group_id": gid,
            "dataset_size": 1000,
        }, headers=auth_headers)
        assert resp.status_code == 200


class TestSystemEndpoints:
    def test_root(self, fresh_client):
        resp = fresh_client.get("/")
        assert resp.status_code == 200
        assert resp.json()["message"] == "Federated Learning API"

    def test_health(self, fresh_client):
        resp = fresh_client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"

    def test_server_status(self, fresh_client, auth_headers):
        resp = fresh_client.get("/api/server/status", headers=auth_headers)
        assert resp.status_code == 200

    def test_system_metrics(self, fresh_client, auth_headers):
        resp = fresh_client.get("/api/system/metrics", headers=auth_headers)
        assert resp.status_code == 200


class TestNotificationsEndpoints:
    def test_unread_count(self, fresh_client, auth_headers):
        resp = fresh_client.get("/api/notifications/unread-count", headers=auth_headers)
        assert resp.status_code == 200
        assert "count" in resp.json()

    def test_list_notifications(self, fresh_client, auth_headers):
        resp = fresh_client.get("/api/notifications", headers=auth_headers)
        assert resp.status_code == 200


class TestLogsEndpoints:
    def test_list_logs(self, fresh_client, auth_headers):
        resp = fresh_client.get("/api/logs?limit=10", headers=auth_headers)
        assert resp.status_code == 200
        assert "logs" in resp.json()

    def test_logs_filter_by_group(self, fresh_client, auth_headers, group_id):
        resp = fresh_client.get(f"/api/logs?group_id={group_id}", headers=auth_headers)
        assert resp.status_code == 200


class TestRecommendationEndpoints:
    def test_unified_recommendation(self, fresh_client, auth_headers):
        resp = fresh_client.post("/api/recommendations/unified", json={
            "dataset_size": 10000,
            "task_type": "image_classification",
            "num_classes": 10,
        }, headers=auth_headers)
        assert resp.status_code == 200
        assert "recommendations" in resp.json()


class TestEdgeCases:
    def test_invalid_json(self, fresh_client):
        resp = fresh_client.post("/api/auth/login", content=b"not json")
        assert resp.status_code == 422

    def test_empty_group_create(self, fresh_client, auth_headers):
        resp = fresh_client.post("/api/groups", json={}, headers=auth_headers)
        assert resp.status_code == 400

    def test_group_create_invalid_window(self, fresh_client, auth_headers):
        mid = f"inv_m_{os.urandom(4).hex()}"
        fresh_client.post("/api/models/register/architecture", json={
            "model_id": mid,
            "architecture_path": "torch.nn.Linear",
            "config": {"in_features": 10, "out_features": 3},
        }, headers=auth_headers)
        resp = fresh_client.post("/api/groups", json={
            "group_id": f"inv_{os.urandom(4).hex()}",
            "model_id": mid,
            "window_size": -1,
        }, headers=auth_headers)
        assert resp.status_code == 400
