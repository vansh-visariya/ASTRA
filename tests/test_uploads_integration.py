"""
End-to-end test: client uploads a delta through the presigned-URL flow,
server validates, aggregates, returns global_version=1.

This exercises the full pipeline:
  init → PUT blob (chunked) → complete → verify → dispatch into FLServer

Uses SimpleMLP (73 params) so the delta is tiny (292 bytes).
"""

import hashlib
import shutil
import tempfile
import uuid
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from astra.app.server_api import app
from astra.app.uploads import LocalDiskObjectStore, UploadManager


@pytest.fixture
def temp_uploads_dir(monkeypatch):
    """Per-test temp uploads dir."""
    tmp = Path(tempfile.mkdtemp(prefix="astra_uploads_test_"))
    secret_key = b"astra-test-secret"
    import astra.app.uploads as um_mod

    store = LocalDiskObjectStore(disk_path=str(tmp), min_free_bytes=0)
    manager = UploadManager(store=store, secret_key=secret_key, presign_ttl=60)
    monkeypatch.setattr(um_mod, "_upload_manager", manager)
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def fresh_client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def admin_creds(fresh_client):
    username = f"int_admin_{uuid.uuid4().hex[:6]}"
    pw = "testpass123"
    fresh_client.post(
        "/api/auth/signup",
        json={"username": username, "password": pw, "role": "admin"},
    )
    r = fresh_client.post(
        "/api/auth/login", json={"username": username, "password": pw}
    )
    return username, pw, r.json().get("token", "")


@pytest.fixture
def client_creds(fresh_client):
    username = f"int_client_{uuid.uuid4().hex[:6]}"
    pw = "testpass123"
    fresh_client.post(
        "/api/auth/signup",
        json={"username": username, "password": pw, "role": "client"},
    )
    r = fresh_client.post(
        "/api/auth/login", json={"username": username, "password": pw}
    )
    return username, pw, r.json().get("token", "")


class TestUploadIntegration:
    def test_full_upload_cycle_via_presigned_url(
        self, fresh_client, admin_creds, client_creds, temp_uploads_dir
    ):
        """Register a SimpleMLP, create a group, activate a client, upload a
        delta through the presigned-URL flow, verify global_version advanced."""
        from astra.core.models.model_zoo import SimpleMLP
        from astra.infra.registry import ModelInfo, get_registry

        admin_user, _admin_pw, admin_tok = admin_creds
        client_user, _client_pw, client_tok = client_creds
        admin_h = {"Authorization": f"Bearer {admin_tok}"}
        client_h = {"Authorization": f"Bearer {client_tok}"}

        # 1. Register SimpleMLP directly in the registry
        model_id = f"upl_mlp_{uuid.uuid4().hex[:6]}"
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

        # 2. Create group + activate client
        gid = f"upl_grp_{uuid.uuid4().hex[:6]}"
        r = fresh_client.post(
            "/api/groups",
            json={"group_id": gid, "model_id": model_id, "window_size": 1, "time_limit": 60.0},
            headers=admin_h,
        )
        assert r.status_code == 200, r.text
        fresh_client.post(f"/api/groups/{gid}/start", headers=admin_h)

        fresh_client.post(
            "/api/join/join-request", json={"group_id": gid}, headers=client_h
        )
        pending = fresh_client.get(
            f"/api/join/join-requests?group_id={gid}", headers=admin_h
        ).json()
        fresh_client.post(
            "/api/join/join-requests/approve",
            json={"request_id": pending["requests"][0]["id"]},
            headers=admin_h,
        )
        act = fresh_client.post(
            f"/api/join/activate/{gid}", headers=client_h
        ).json()
        cid = act["client_id"]

        # 3. Build the delta (73 floats for SimpleMLP)
        n_params = sum(p.numel() for p in SimpleMLP().parameters())
        delta = np.random.default_rng(42).standard_normal(n_params).astype("<f4")
        delta_bytes = delta.tobytes()
        sha = hashlib.sha256(delta_bytes).hexdigest()

        # 4. Init upload
        r = fresh_client.post(
            "/api/uploads/init",
            json={
                "client_id": cid,
                "group_id": gid,
                "content_length": len(delta_bytes),
                "sha256": sha,
            },
            headers=client_h,
        )
        assert r.status_code == 200, r.text
        init_body = r.json()
        upload_id = init_body["upload_id"]
        upload_url = init_body["upload_url"]

        # 5. PUT the bytes — single PUT (no chunking needed for 292 bytes)
        r = fresh_client.put(upload_url, content=delta_bytes)
        assert r.status_code == 200, r.text
        put_body = r.json()
        assert put_body["received"] == len(delta_bytes)
        assert put_body["complete"] is True

        # 6. Complete — verifies sha256 + dispatches into the FLServer
        r = fresh_client.post(
            f"/api/uploads/{upload_id}/complete",
            json={"sha256": sha},
            headers=client_h,
        )
        assert r.status_code == 200, r.text
        complete_body = r.json()
        assert complete_body["status"] == "completed"
        assert complete_body["sha256"] == sha
        assert complete_body["size"] == len(delta_bytes)
        assert complete_body["global_version"] == 1

        # 7. Verify the group state reflects a completed round
        gs = fresh_client.get(f"/api/groups/{gid}", headers=admin_h).json()
        assert gs["group"]["model_version"] == 1
        assert gs["group"]["completed_rounds"] == 1

        # 8. Verify the disk was cleaned up
        manager_dir = Path(temp_uploads_dir)
        blob_files = list(manager_dir.glob("*.bin"))
        meta_files = list(manager_dir.glob("*.meta.json"))
        assert len(blob_files) == 0
        assert len(meta_files) == 0

    def test_upload_rejects_wrong_sha256(
        self, fresh_client, admin_creds, client_creds, temp_uploads_dir
    ):
        """If the uploaded bytes' sha256 doesn't match the declared sha256, complete fails."""
        from astra.core.models.model_zoo import SimpleMLP
        from astra.infra.registry import ModelInfo, get_registry

        admin_user, _pw, admin_tok = admin_creds
        client_user, _pw, client_tok = client_creds
        admin_h = {"Authorization": f"Bearer {admin_tok}"}
        client_h = {"Authorization": f"Bearer {client_tok}"}

        # Set up: model + group + activated client (same as above but compressed)
        model_id = f"sha_mlp_{uuid.uuid4().hex[:6]}"
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

        gid = f"sha_grp_{uuid.uuid4().hex[:6]}"
        fresh_client.post(
            "/api/groups",
            json={"group_id": gid, "model_id": model_id, "window_size": 1, "time_limit": 60.0},
            headers=admin_h,
        )
        fresh_client.post(f"/api/groups/{gid}/start", headers=admin_h)
        fresh_client.post(
            "/api/join/join-request", json={"group_id": gid}, headers=client_h
        )
        pending = fresh_client.get(
            f"/api/join/join-requests?group_id={gid}", headers=admin_h
        ).json()
        fresh_client.post(
            "/api/join/join-requests/approve",
            json={"request_id": pending["requests"][0]["id"]},
            headers=admin_h,
        )
        act = fresh_client.post(
            f"/api/join/activate/{gid}", headers=client_h
        ).json()
        cid = act["client_id"]

        # Build delta
        n = sum(p.numel() for p in SimpleMLP().parameters())
        delta_bytes = np.zeros(n, dtype="<f4").tobytes()
        wrong_sha = "0" * 64  # wrong sha
        right_sha = hashlib.sha256(delta_bytes).hexdigest()

        r = fresh_client.post(
            "/api/uploads/init",
            json={"client_id": cid, "content_length": len(delta_bytes), "sha256": wrong_sha},
            headers=client_h,
        )
        assert r.status_code == 200
        upload_id = r.json()["upload_id"]
        url = r.json()["upload_url"]
        fresh_client.put(url, content=delta_bytes)

        # Complete with the wrong sha — should reject
        r = fresh_client.post(
            f"/api/uploads/{upload_id}/complete",
            json={"sha256": right_sha},
            headers=client_h,
        )
        assert r.status_code == 400
        assert "sha256" in r.json()["detail"].lower()

    def test_upload_with_wrong_size_rejected(
        self, fresh_client, admin_creds, client_creds, temp_uploads_dir
    ):
        """If the delta doesn't match the model's param count, complete fails."""
        from astra.core.models.model_zoo import SimpleMLP
        from astra.infra.registry import ModelInfo, get_registry

        admin_user, _pw, admin_tok = admin_creds
        client_user, _pw, client_tok = client_creds
        admin_h = {"Authorization": f"Bearer {admin_tok}"}
        client_h = {"Authorization": f"Bearer {client_tok}"}

        model_id = f"size_mlp2_{uuid.uuid4().hex[:6]}"
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

        gid = f"size_grp2_{uuid.uuid4().hex[:6]}"
        fresh_client.post(
            "/api/groups",
            json={"group_id": gid, "model_id": model_id, "window_size": 1, "time_limit": 60.0},
            headers=admin_h,
        )
        fresh_client.post(f"/api/groups/{gid}/start", headers=admin_h)
        fresh_client.post(
            "/api/join/join-request", json={"group_id": gid}, headers=client_h
        )
        pending = fresh_client.get(
            f"/api/join/join-requests?group_id={gid}", headers=admin_h
        ).json()
        fresh_client.post(
            "/api/join/join-requests/approve",
            json={"request_id": pending["requests"][0]["id"]},
            headers=admin_h,
        )
        act = fresh_client.post(
            f"/api/join/activate/{gid}", headers=client_h
        ).json()
        cid = act["client_id"]

        # Send a wrong-size delta (10 floats instead of 73)
        wrong_size_bytes = np.random.default_rng(0).standard_normal(10).astype("<f4").tobytes()
        sha = hashlib.sha256(wrong_size_bytes).hexdigest()
        r = fresh_client.post(
            "/api/uploads/init",
            json={"client_id": cid, "content_length": len(wrong_size_bytes), "sha256": sha},
            headers=client_h,
        )
        assert r.status_code == 200
        upload_id = r.json()["upload_id"]
        url = r.json()["upload_url"]
        fresh_client.put(url, content=wrong_size_bytes)

        r = fresh_client.post(
            f"/api/uploads/{upload_id}/complete",
            json={"sha256": sha},
            headers=client_h,
        )
        assert r.status_code == 400
        assert "size" in r.json()["detail"].lower() or "parameters" in r.json()["detail"].lower()
