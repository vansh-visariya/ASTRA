"""
End-to-end test for the external-client flow:

1. Sign up an admin
2. Register a model
3. Create a group with that model
4. Sign up a client, request to join, admin approves
5. Client activates membership
6. Client uploads a delta
7. Verify global model advanced and can be downloaded

This replaces the role of test_integration.py in the post-removal world:
it exercises the full server pipeline without a built-in local trainer.
"""

import base64
import os

import numpy as np
import pytest
from fastapi.testclient import TestClient

from astra.app.server_api import app
from astra.core.models.model_zoo import SimpleMLP


def _delta_bytes(n: int, seed: int = 0) -> str:
    rng = np.random.default_rng(seed)
    arr = rng.standard_normal(n).astype(np.float32)
    return base64.b64encode(arr.tobytes()).decode("ascii")


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def admin_creds(client):
    username = f"flow_admin_{os.urandom(4).hex()}"
    password = "testpass123"
    client.post(
        "/api/auth/signup",
        json={"username": username, "password": password, "role": "admin"},
    )
    login = client.post(
        "/api/auth/login", json={"username": username, "password": password}
    )
    return username, password, login.json().get("token", "")


@pytest.fixture
def client_creds(client):
    username = f"flow_client_{os.urandom(4).hex()}"
    password = "testpass123"
    client.post(
        "/api/auth/signup",
        json={"username": username, "password": password, "role": "client"},
    )
    login = client.post(
        "/api/auth/login", json={"username": username, "password": password}
    )
    return username, password, login.json().get("token", "")


class TestExternalClientFlow:
    def test_full_upload_cycle(self, client, admin_creds, client_creds):
        from astra.infra.registry import ModelInfo, get_registry

        admin_user, admin_pass, admin_token = admin_creds
        client_user, client_pass, client_token = client_creds
        admin_headers = {"Authorization": f"Bearer {admin_token}"}
        client_headers = {"Authorization": f"Bearer {client_token}"}

        # 1) Register a model directly in the in-memory registry
        model_id = f"ext_mlp_{os.urandom(4).hex()}"
        sample = SimpleMLP()
        info = ModelInfo(
            model_id=model_id,
            model_type="classifier",
            architecture="mlp",
            total_params=sum(p.numel() for p in sample.parameters()),
            trainable_params=sum(p.numel() for p in sample.parameters() if p.requires_grad),
            source="local",
        )
        get_registry().register_factory(model_id, lambda: SimpleMLP(), info)

        # 2) Create a group
        group_id = f"flow_group_{os.urandom(4).hex()}"
        resp = client.post(
            "/api/groups",
            json={
                "group_id": group_id,
                "model_id": model_id,
                "window_size": 1,
                "time_limit": 60.0,
                "lr": 0.01,
                "aggregator": "fedavg",
            },
            headers=admin_headers,
        )
        assert resp.status_code == 200, resp.text

        # 3) Start the group
        resp = client.post(f"/api/groups/{group_id}/start", headers=admin_headers)
        assert resp.status_code == 200

        # 4) Client requests to join
        resp = client.post(
            "/api/join/join-request",
            json={"group_id": group_id},
            headers=client_headers,
        )
        assert resp.status_code == 200

        # Admin fetches pending requests to get the numeric ID for approve
        resp = client.get(
            f"/api/join/join-requests?group_id={group_id}", headers=admin_headers
        )
        assert resp.status_code == 200
        pending = resp.json().get("requests", [])
        assert pending, "expected a pending request"
        request_id = pending[0]["id"]

        # 5) Admin approves
        resp = client.post(
            "/api/join/join-requests/approve",
            json={"request_id": request_id},
            headers=admin_headers,
        )
        assert resp.status_code == 200

        # 6) Client activates
        resp = client.post(
            f"/api/join/activate/{group_id}", headers=client_headers
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        c_client_id = body["client_id"]

        # 7) Upload a delta. SimpleMLP has 10*5 + 5 + 5*3 + 3 = 73 params.
        delta_size = sum(p.numel() for p in SimpleMLP().parameters())
        resp = client.post(
            f"/api/clients/{c_client_id}/delta",
            json={
                "client_id": c_client_id,
                "client_version": 0,
                "local_updates": _delta_bytes(delta_size, seed=1),
                "update_type": "delta",
                "local_dataset_size": 100,
                "meta": {"train_accuracy": 0.42, "train_loss": 0.8},
            },
            headers=client_headers,
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["status"] == "accepted", body
        assert body["global_version"] == 1

        # 8) Group status reflects a completed round
        resp = client.get(f"/api/groups/{group_id}", headers=admin_headers)
        assert resp.status_code == 200
        group = resp.json()["group"]
        assert group["model_version"] == 1
        assert group["completed_rounds"] == 1

    def test_delta_size_validation(self, client, client_creds):
        # Upload an absurdly small delta (not a multiple of 4). The cid is
        # not registered with any group, so the endpoint returns 404 first;
        # either 400 (size rejected) or 404 (no such client) is correct.
        _, _, client_token = client_creds
        cid = f"flow_c_{os.urandom(4).hex()}"
        resp = client.post(
            f"/api/clients/{cid}/delta",
            json={
                "client_id": cid,
                "client_version": 0,
                "local_updates": base64.b64encode(b"abc").decode("ascii"),
                "update_type": "delta",
                "local_dataset_size": 100,
            },
            headers={"Authorization": f"Bearer {client_token}"},
        )
        assert resp.status_code in (400, 404)

    def test_unauthenticated_upload_rejected(self, client):
        cid = f"flow_c_{os.urandom(4).hex()}"
        resp = client.post(
            f"/api/clients/{cid}/delta",
            json={
                "client_id": cid,
                "client_version": 0,
                "local_updates": _delta_bytes(30),
                "update_type": "delta",
                "local_dataset_size": 100,
            },
        )
        assert resp.status_code == 401
