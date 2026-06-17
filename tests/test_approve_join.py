"""Quick sanity test for the approve-join-request endpoint fix."""
import os
import uuid

import pytest
from fastapi.testclient import TestClient

from astra.app.server_api import app


@pytest.fixture
def fresh_client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def admin_creds(fresh_client):
    user = f"approve_admin_{uuid.uuid4().hex[:8]}"
    fresh_client.post(
        "/api/auth/signup",
        json={"username": user, "password": "testpass123", "role": "admin"},
    )
    r = fresh_client.post(
        "/api/auth/login", json={"username": user, "password": "testpass123"}
    ).json()
    return user, r["token"]


@pytest.fixture
def client_creds(fresh_client):
    user = f"approve_client_{uuid.uuid4().hex[:8]}"
    fresh_client.post(
        "/api/auth/signup",
        json={"username": user, "password": "testpass123", "role": "client"},
    )
    r = fresh_client.post(
        "/api/auth/login", json={"username": user, "password": "testpass123"}
    ).json()
    return user, r["token"]


def test_approve_join_request_returns_token(fresh_client, admin_creds, client_creds):
    admin_user, admin_tok = admin_creds
    client_user, client_tok = client_creds
    admin_h = {"Authorization": f"Bearer {admin_tok}"}
    client_h = {"Authorization": f"Bearer {client_tok}"}

    # Create group with a registered model
    gid = f"approve_{uuid.uuid4().hex[:8]}"
    fresh_client.post(
        "/api/models/register/architecture",
        json={
            "model_id": "approve_test_mlp",
            "architecture_path": "astra.core.models.model_zoo.SimpleMLP",
            "model_type": "vision",
            "config": {},
        },
        headers=admin_h,
    )
    fresh_client.post(
        "/api/groups",
        json={"group_id": gid, "model_id": "approve_test_mlp"},
        headers=admin_h,
    )

    # Client requests to join
    r = fresh_client.post(
        "/api/join/join-request", json={"group_id": gid}, headers=client_h
    )
    assert r.status_code == 200, r.text

    # Admin approves — this is the bug we just fixed
    pending = fresh_client.get(
        f"/api/join/join-requests?group_id={gid}", headers=admin_h
    ).json()
    request_id = pending["requests"][0]["id"]
    r = fresh_client.post(
        "/api/join/join-requests/approve",
        json={"request_id": request_id},
        headers=admin_h,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body.get("success") is True
    assert "token" in body
    assert len(body["token"]) > 0

    # Idempotent: approve again — should also succeed with already_approved=True
    r = fresh_client.post(
        "/api/join/join-requests/approve",
        json={"request_id": request_id},
        headers=admin_h,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body.get("success") is True
    assert body.get("already_approved") is True
