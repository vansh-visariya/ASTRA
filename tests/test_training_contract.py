"""
Tests for the Training Contract system.

Covers: TrainingManifest validation, contract versioning,
delta upload with optional expected_delta_bytes, and validation data upload.
"""

import base64
import os

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from astra.app.server_api import app
from astra.core.models.model_zoo import SimpleMLP
from astra.infra.models import TrainingManifest


@pytest.fixture
def fresh_client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def admin_creds(fresh_client):
    username = f"tc_admin_{os.urandom(4).hex()}"
    password = "testpass123"
    fresh_client.post("/api/auth/signup", json={
        "username": username, "password": password, "role": "admin",
    })
    login = fresh_client.post("/api/auth/login", json={
        "username": username, "password": password,
    })
    return username, password, login.json().get("token", "")


@pytest.fixture
def client_creds(fresh_client):
    username = f"tc_client_{os.urandom(4).hex()}"
    password = "testpass123"
    fresh_client.post("/api/auth/signup", json={
        "username": username, "password": password, "role": "client",
    })
    login = fresh_client.post("/api/auth/login", json={
        "username": username, "password": password,
    })
    return username, password, login.json().get("token", "")


def _register_model_registry(mid):
    from astra.infra.registry import ModelInfo, get_registry
    sample = SimpleMLP()
    info = ModelInfo(
        model_id=mid,
        model_type="classifier",
        architecture="mlp",
        total_params=sum(p.numel() for p in sample.parameters()),
        trainable_params=sum(p.numel() for p in sample.parameters() if p.requires_grad),
        source="local",
    )
    get_registry().register_factory(mid, lambda: SimpleMLP(), info)


def _delta_bytes(n: int, seed: int = 0) -> str:
    rng = np.random.default_rng(seed)
    arr = rng.standard_normal(n).astype(np.float32)
    return base64.b64encode(arr.tobytes()).decode("ascii")


# ======================================================================
# TrainingManifest Pydantic model tests
# ======================================================================

class TestTrainingManifestModel:
    def test_manifest_optional_expected_delta_bytes(self):
        m = TrainingManifest(model_id="test_model")
        assert m.expected_delta_bytes is None

    def test_manifest_contract_version_default(self):
        m = TrainingManifest(model_id="test_model")
        assert m.contract_version == 1

    def test_manifest_with_all_fields(self):
        m = TrainingManifest(
            model_id="test_model",
            is_peft=True,
            target_modules=["q_proj", "v_proj"],
            lora_rank=8,
            lora_alpha=16.0,
            expected_delta_bytes=4096,
            lr=0.001,
            batch_size=16,
            local_epochs=5,
            optimizer="adam",
            loss_function="mse",
            max_grad_norm=1.0,
            input_shape=[784],
            num_classes=10,
            label_type="classification",
            data_description="Test data",
            preprocessing_steps=["normalize"],
            accepted_update_types=["delta"],
            val_metric="accuracy",
            contract_version=2,
        )
        assert m.contract_version == 2
        assert m.expected_delta_bytes == 4096
        assert m.data_description == "Test data"
        assert m.preprocessing_steps == ["normalize"]

    def test_manifest_model_dump_excludes_none(self):
        m = TrainingManifest(model_id="test_model")
        d = m.model_dump()
        assert d["model_id"] == "test_model"
        assert d["expected_delta_bytes"] is None
        assert d["contract_version"] == 1


# ======================================================================
# Contract versioning on manifest update
# ======================================================================

class TestContractVersioning:
    def test_create_group_with_manifest(self, fresh_client, admin_creds):
        _, _, admin_token = admin_creds
        admin_headers = {"Authorization": f"Bearer {admin_token}"}
        mid = f"cv_model_{os.urandom(4).hex()}"
        _register_model_registry(mid)
        gid = f"cv_group_{os.urandom(4).hex()}"
        manifest = {"model_id": mid, "lr": 0.01, "batch_size": 32}
        resp = fresh_client.post("/api/groups", json={
            "group_id": gid, "model_id": mid, "window_size": 3, "time_limit": 20,
            "training_manifest": manifest,
        }, headers=admin_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["group"]["training_manifest"]["contract_version"] == 1

    def test_update_manifest_bumps_version(self, fresh_client, admin_creds):
        _, _, admin_token = admin_creds
        admin_headers = {"Authorization": f"Bearer {admin_token}"}
        mid = f"cv_model_{os.urandom(4).hex()}"
        _register_model_registry(mid)
        gid = f"cv_group_{os.urandom(4).hex()}"
        manifest = {"model_id": mid, "lr": 0.01}
        fresh_client.post("/api/groups", json={
            "group_id": gid, "model_id": mid, "window_size": 3, "time_limit": 20,
            "training_manifest": manifest,
        }, headers=admin_headers)

        updated = {"model_id": mid, "lr": 0.001, "batch_size": 64}
        resp = fresh_client.put(
            f"/api/groups/{gid}/manifest", json=updated, headers=admin_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["manifest"]["contract_version"] == 2
        assert data["manifest"]["lr"] == 0.001

    def test_update_manifest_invalid_returns_400(self, fresh_client, admin_creds):
        _, _, admin_token = admin_creds
        admin_headers = {"Authorization": f"Bearer {admin_token}"}
        mid = f"cv_model_{os.urandom(4).hex()}"
        _register_model_registry(mid)
        gid = f"cv_group_{os.urandom(4).hex()}"
        fresh_client.post("/api/groups", json={
            "group_id": gid, "model_id": mid, "window_size": 3, "time_limit": 20,
        }, headers=admin_headers)

        resp = fresh_client.put(
            f"/api/groups/{gid}/manifest", json={"lr": 0.001}, headers=admin_headers,
        )
        assert resp.status_code == 400


# ======================================================================
# Delta upload without expected_delta_bytes
# ======================================================================

class TestDeltaUploadWithoutExpectedBytes:
    def test_upload_without_expected_bytes_succeeds(self, fresh_client, admin_creds, client_creds):
        _, _, admin_token = admin_creds
        _, _, client_token = client_creds
        admin_headers = {"Authorization": f"Bearer {admin_token}"}
        client_headers = {"Authorization": f"Bearer {client_token}"}

        mid = f"du_model_{os.urandom(4).hex()}"
        _register_model_registry(mid)
        gid = f"du_group_{os.urandom(4).hex()}"

        # Create group WITHOUT expected_delta_bytes
        resp = fresh_client.post("/api/groups", json={
            "group_id": gid, "model_id": mid, "window_size": 1, "time_limit": 60,
            "training_manifest": {"model_id": mid, "lr": 0.01},
        }, headers=admin_headers)
        assert resp.status_code == 200

        # Start group
        resp = fresh_client.post(f"/api/groups/{gid}/start", headers=admin_headers)
        assert resp.status_code == 200

        # Client requests to join
        resp = fresh_client.post("/api/join/join-request", json={"group_id": gid}, headers=client_headers)
        assert resp.status_code == 200

        # Admin approves
        resp = fresh_client.get(f"/api/join/join-requests?group_id={gid}", headers=admin_headers)
        pending = resp.json().get("requests", [])
        assert pending
        resp = fresh_client.post("/api/join/join-requests/approve", json={"request_id": pending[0]["id"]}, headers=admin_headers)
        assert resp.status_code == 200

        # Client activates
        resp = fresh_client.post(f"/api/join/activate/{gid}", headers=client_headers)
        assert resp.status_code == 200
        cid = resp.json()["client_id"]

        # Upload delta
        delta_size = sum(p.numel() for p in SimpleMLP().parameters())
        resp = fresh_client.post(
            f"/api/clients/{cid}/delta",
            json={
                "client_id": cid,
                "client_version": 0,
                "local_updates": _delta_bytes(delta_size, seed=1),
                "update_type": "delta",
                "local_dataset_size": 100,
            },
            headers=client_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "accepted"
        assert data.get("contract_version") == 1

    def test_upload_nan_still_rejected(self, fresh_client, admin_creds, client_creds):
        _, _, admin_token = admin_creds
        _, _, client_token = client_creds
        admin_headers = {"Authorization": f"Bearer {admin_token}"}
        client_headers = {"Authorization": f"Bearer {client_token}"}

        mid = f"du_model_{os.urandom(4).hex()}"
        _register_model_registry(mid)
        gid = f"du_group_{os.urandom(4).hex()}"
        fresh_client.post("/api/groups", json={
            "group_id": gid, "model_id": mid, "window_size": 1, "time_limit": 60,
        }, headers=admin_headers)
        fresh_client.post(f"/api/groups/{gid}/start", headers=admin_headers)

        resp = fresh_client.post("/api/join/join-request", json={"group_id": gid}, headers=client_headers)
        resp = fresh_client.get(f"/api/join/join-requests?group_id={gid}", headers=admin_headers)
        pending = resp.json().get("requests", [])
        fresh_client.post("/api/join/join-requests/approve", json={"request_id": pending[0]["id"]}, headers=admin_headers)
        resp = fresh_client.post(f"/api/join/activate/{gid}", headers=client_headers)
        cid = resp.json()["client_id"]

        # Delta with NaN — 10 params to match SimpleMLP
        delta = np.ones(10, dtype=np.float32)
        delta[5] = float('nan')
        delta_b64 = base64.b64encode(delta.tobytes()).decode()
        resp = fresh_client.post(
            f"/api/clients/{cid}/delta",
            json={
                "client_id": cid,
                "client_version": 0,
                "local_updates": delta_b64,
                "update_type": "delta",
                "local_dataset_size": 100,
            },
            headers=client_headers,
        )
        assert resp.status_code == 400


# ======================================================================
# Validation data upload
# ======================================================================

class TestValidationDataUpload:
    def test_upload_valid_pt_file(self, fresh_client, admin_creds):
        _, _, admin_token = admin_creds
        admin_headers = {"Authorization": f"Bearer {admin_token}"}
        mid = f"vd_model_{os.urandom(4).hex()}"
        _register_model_registry(mid)
        gid = f"vd_group_{os.urandom(4).hex()}"
        fresh_client.post("/api/groups", json={
            "group_id": gid, "model_id": mid, "window_size": 3, "time_limit": 20,
        }, headers=admin_headers)

        X = torch.randn(100, 10)
        y = torch.randint(0, 3, (100,))
        pt_path = f"/tmp/test_val_{os.urandom(4).hex()}.pt"
        torch.save({"X": X, "y": y}, pt_path)

        try:
            with open(pt_path, "rb") as f:
                resp = fresh_client.post(
                    f"/api/groups/{gid}/validation-data",
                    files={"file": ("val_data.pt", f, "application/octet-stream")},
                    headers=admin_headers,
                )
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "uploaded"
            assert data["group_id"] == gid
            assert "val_data.pt" in data["val_dataset"]
        finally:
            os.remove(pt_path)

    def test_upload_invalid_format_rejected(self, fresh_client, admin_creds):
        _, _, admin_token = admin_creds
        admin_headers = {"Authorization": f"Bearer {admin_token}"}
        mid = f"vd_model_{os.urandom(4).hex()}"
        _register_model_registry(mid)
        gid = f"vd_group_{os.urandom(4).hex()}"
        fresh_client.post("/api/groups", json={
            "group_id": gid, "model_id": mid, "window_size": 3, "time_limit": 20,
        }, headers=admin_headers)

        pt_path = f"/tmp/test_val_{os.urandom(4).hex()}.pt"
        torch.save({"wrong_key": torch.randn(10, 10)}, pt_path)
        try:
            with open(pt_path, "rb") as f:
                resp = fresh_client.post(
                    f"/api/groups/{gid}/validation-data",
                    files={"file": ("val_data.pt", f, "application/octet-stream")},
                    headers=admin_headers,
                )
            assert resp.status_code == 400
        finally:
            os.remove(pt_path)

    def test_upload_non_pt_rejected(self, fresh_client, admin_creds):
        _, _, admin_token = admin_creds
        admin_headers = {"Authorization": f"Bearer {admin_token}"}
        mid = f"vd_model_{os.urandom(4).hex()}"
        _register_model_registry(mid)
        gid = f"vd_group_{os.urandom(4).hex()}"
        fresh_client.post("/api/groups", json={
            "group_id": gid, "model_id": mid, "window_size": 3, "time_limit": 20,
        }, headers=admin_headers)

        resp = fresh_client.post(
            f"/api/groups/{gid}/validation-data",
            files={"file": ("data.csv", b"a,b,c", "text/csv")},
            headers=admin_headers,
        )
        assert resp.status_code == 400

    def test_upload_no_auth_returns_401(self, fresh_client):
        resp = fresh_client.post(
            "/api/groups/nonexistent/validation-data",
            files={"file": ("val_data.pt", b"fake", "application/octet-stream")},
        )
        assert resp.status_code == 401
