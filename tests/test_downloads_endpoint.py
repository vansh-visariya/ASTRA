"""
Tests for the chunked download flow (/api/downloads/*).

Exercises the full lifecycle: init → sign chunk URLs → GET chunks →
verify reassembly + sha256 → complete. Uses a real model file on disk
written by `GroupManager.save_model_weights` via a temporary group.
"""

import hashlib
import os
import shutil
import tempfile
import uuid
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from astra.app.server_api import app


@pytest.fixture
def temp_models_dir(monkeypatch):
    """Redirect GroupManager to a per-test models/ dir."""
    tmp = Path(tempfile.mkdtemp(prefix="astra_dl_models_"))
    cwd = os.getcwd()
    os.chdir(tmp)
    yield tmp
    os.chdir(cwd)
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def fresh_client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def admin_creds(fresh_client):
    username = f"dl_admin_{uuid.uuid4().hex[:6]}"
    pw = "testpass123"
    fresh_client.post(
        "/api/auth/signup",
        json={"username": username, "password": pw, "role": "admin"},
    )
    r = fresh_client.post(
        "/api/auth/login", json={"username": username, "password": pw}
    )
    return username, pw, r.json()["token"]


def _seed_group_with_model(
    fresh_client, admin_h, group_id: str, model_id: str, n_params: int = 73
):
    """Create a group, register SimpleMLP, force-save a checkpoint with
    `n_params` weights so the download endpoint has something to serve."""
    from astra.core.models.model_zoo import SimpleMLP
    from astra.infra.registry import ModelInfo, get_registry

    info = ModelInfo(
        model_id=model_id,
        model_type="classifier",
        architecture="mlp",
        total_params=n_params,
        trainable_params=n_params,
        source="local",
    )
    get_registry().register_factory(model_id, lambda: SimpleMLP(), info)

    r = fresh_client.post(
        "/api/groups",
        json={
            "group_id": group_id,
            "model_id": model_id,
            "window_size": 1,
            "time_limit": 60.0,
        },
        headers=admin_h,
    )
    assert r.status_code == 200, r.text

    save_dir = Path("models") / "global" / group_id
    save_dir.mkdir(parents=True, exist_ok=True)
    weights = np.random.default_rng(0).standard_normal(n_params).astype("<f4")
    ckpt_path = save_dir / "model_latest.pt"
    import torch

    torch.save(
        {
            "version": 1,
            "weights": weights,
            "accuracy": 0.5,
            "loss": 1.0,
            "num_clients": 1,
            "timestamp": 0.0,
            "group_id": group_id,
        },
        ckpt_path,
    )


class TestDownloadInit:
    def test_init_requires_auth(self, fresh_client):
        r = fresh_client.post(
            "/api/downloads/init", json={"group_id": "x"}
        )
        assert r.status_code == 401

    def test_init_404_for_missing_group(self, fresh_client, admin_creds):
        admin_h = {"Authorization": f"Bearer {admin_creds[2]}"}
        r = fresh_client.post(
            "/api/downloads/init",
            json={"group_id": "nope_no_such_group"},
            headers=admin_h,
        )
        assert r.status_code == 404

    def test_init_404_for_missing_model_file(
        self, fresh_client, admin_creds, temp_models_dir
    ):
        admin_h = {"Authorization": f"Bearer {admin_creds[2]}"}
        gid = f"empty_grp_{uuid.uuid4().hex[:6]}"
        from astra.core.models.model_zoo import SimpleMLP
        from astra.infra.registry import ModelInfo, get_registry

        model_id = f"empty_mlp_{uuid.uuid4().hex[:6]}"
        get_registry().register_factory(
            model_id,
            lambda: SimpleMLP(),
            ModelInfo(
                model_id=model_id,
                model_type="classifier",
                architecture="mlp",
                total_params=73,
                trainable_params=73,
                source="local",
            ),
        )
        fresh_client.post(
            "/api/groups",
            json={"group_id": gid, "model_id": model_id, "window_size": 1, "time_limit": 60.0},
            headers=admin_h,
        )
        # Group exists but no .pt file written yet
        r = fresh_client.post(
            "/api/downloads/init",
            json={"group_id": gid, "format": "pt"},
            headers=admin_h,
        )
        assert r.status_code == 404
        assert "no saved models" in r.json()["detail"].lower()

    def test_init_returns_manifest_with_signed_chunk_urls(
        self, fresh_client, admin_creds, temp_models_dir
    ):
        admin_h = {"Authorization": f"Bearer {admin_creds[2]}"}
        gid = f"manifest_grp_{uuid.uuid4().hex[:6]}"
        mid = f"manifest_mlp_{uuid.uuid4().hex[:6]}"
        _seed_group_with_model(fresh_client, admin_h, gid, mid, n_params=73)

        r = fresh_client.post(
            "/api/downloads/init",
            json={"group_id": gid, "format": "pt", "chunk_size": 200},
            headers=admin_h,
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["group_id"] == gid
        assert body["format"] == "pt"
        assert body["chunk_size"] == 200
        assert body["num_chunks"] >= 1
        assert len(body["chunks"]) == body["num_chunks"]
        for i, chunk in enumerate(body["chunks"]):
            assert chunk["index"] == i
            assert "expires=" in chunk["url"]
            assert "sig=" in chunk["url"]

        # sha256 should match the actual file
        ckpt = Path("models") / "global" / gid / "model_latest.pt"
        expected = hashlib.sha256(ckpt.read_bytes()).hexdigest()
        assert body["sha256"] == expected

    def test_init_validates_format(
        self, fresh_client, admin_creds, temp_models_dir
    ):
        admin_h = {"Authorization": f"Bearer {admin_creds[2]}"}
        gid = f"badfmt_grp_{uuid.uuid4().hex[:6]}"
        mid = f"badfmt_mlp_{uuid.uuid4().hex[:6]}"
        _seed_group_with_model(fresh_client, admin_h, gid, mid, n_params=73)
        r = fresh_client.post(
            "/api/downloads/init",
            json={"group_id": gid, "format": "bogus"},
            headers=admin_h,
        )
        assert r.status_code == 400


class TestDownloadChunk:
    def test_get_chunk_with_valid_signature(
        self, fresh_client, admin_creds, temp_models_dir
    ):
        admin_h = {"Authorization": f"Bearer {admin_creds[2]}"}
        gid = f"chunk_grp_{uuid.uuid4().hex[:6]}"
        mid = f"chunk_mlp_{uuid.uuid4().hex[:6]}"
        _seed_group_with_model(fresh_client, admin_h, gid, mid, n_params=73)

        # Init with a chunk size that forces multiple chunks
        ckpt_path = Path("models") / "global" / gid / "model_latest.pt"
        total_size = ckpt_path.stat().st_size
        chunk_size = total_size // 4
        r = fresh_client.post(
            "/api/downloads/init",
            json={"group_id": gid, "format": "pt", "chunk_size": chunk_size},
            headers=admin_h,
        )
        assert r.status_code == 200
        manifest = r.json()
        assert manifest["num_chunks"] >= 4

        # Download all chunks and reassemble
        reassembled = b""
        for i, chunk in enumerate(manifest["chunks"]):
            r = fresh_client.get(chunk["url"])
            assert r.status_code == 200, r.text
            assert r.headers["X-Chunk-Index"] == str(i)
            assert r.headers["X-Total-Sha256"] == manifest["sha256"]
            assert r.headers["X-Num-Chunks"] == str(manifest["num_chunks"])
            assert int(r.headers["Content-Length"]) == len(r.content)
            reassembled += r.content

        assert len(reassembled) == total_size
        assert hashlib.sha256(reassembled).hexdigest() == manifest["sha256"]

    def test_get_chunk_rejects_invalid_signature(
        self, fresh_client, admin_creds, temp_models_dir
    ):
        admin_h = {"Authorization": f"Bearer {admin_creds[2]}"}
        gid = f"badsig_grp_{uuid.uuid4().hex[:6]}"
        mid = f"badsig_mlp_{uuid.uuid4().hex[:6]}"
        _seed_group_with_model(fresh_client, admin_h, gid, mid, n_params=73)

        r = fresh_client.post(
            "/api/downloads/init",
            json={"group_id": gid, "format": "pt"},
            headers=admin_h,
        )
        manifest = r.json()
        url = manifest["chunks"][0]["url"]
        # Tamper with the signature
        tampered = url.replace("sig=", "sig=00000000000000000000000000000000")
        r = fresh_client.get(tampered)
        assert r.status_code == 403

    def test_get_chunk_404_for_unknown_download(
        self, fresh_client
    ):
        r = fresh_client.get(
            "/api/downloads/nonexistent_id/chunk/0?expires=9999999999&sig=deadbeef"
        )
        assert r.status_code in (403, 404)


class TestDownloadComplete:
    def test_complete_returns_stats(
        self, fresh_client, admin_creds, temp_models_dir
    ):
        admin_h = {"Authorization": f"Bearer {admin_creds[2]}"}
        gid = f"comp_grp_{uuid.uuid4().hex[:6]}"
        mid = f"comp_mlp_{uuid.uuid4().hex[:6]}"
        _seed_group_with_model(fresh_client, admin_h, gid, mid, n_params=73)

        r = fresh_client.post(
            "/api/downloads/init",
            json={"group_id": gid, "format": "pt"},
            headers=admin_h,
        )
        manifest = r.json()
        did = manifest["download_id"]

        # Serve at least one chunk so bytes_served > 0
        fresh_client.get(manifest["chunks"][0]["url"])

        r = fresh_client.post(
            f"/api/downloads/{did}/complete", headers=admin_h
        )
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "completed"
        assert body["sha256"] == manifest["sha256"]
        assert body["total_size"] == manifest["total_size"]
        assert body["bytes_served"] == manifest["total_size"]

    def test_complete_404_for_unknown(
        self, fresh_client, admin_creds
    ):
        admin_h = {"Authorization": f"Bearer {admin_creds[2]}"}
        r = fresh_client.post(
            "/api/downloads/nonexistent_id/complete", headers=admin_h
        )
        assert r.status_code == 404


class TestDownloadAbort:
    def test_abort_frees_slot(
        self, fresh_client, admin_creds, temp_models_dir
    ):
        admin_h = {"Authorization": f"Bearer {admin_creds[2]}"}
        gid = f"abort_grp_{uuid.uuid4().hex[:6]}"
        mid = f"abort_mlp_{uuid.uuid4().hex[:6]}"
        _seed_group_with_model(fresh_client, admin_h, gid, mid, n_params=73)

        r = fresh_client.post(
            "/api/downloads/init",
            json={"group_id": gid, "format": "pt"},
            headers=admin_h,
        )
        did = r.json()["download_id"]

        r = fresh_client.delete(
            f"/api/downloads/{did}", headers=admin_h
        )
        assert r.status_code == 200

        # GET on the aborted slot should now 404
        r = fresh_client.get(f"/api/downloads/{did}", headers=admin_h)
        assert r.status_code == 404


class TestDownloadEndToEnd:
    def test_client_can_download_full_model_with_progress(
        self, fresh_client, admin_creds, temp_models_dir
    ):
        """Simulates what the dashboard client would do:
        init → fetch each chunk → reassemble → verify sha256 → POST /complete."""
        import requests  # noqa: F401  (verifies pattern works with requests lib)

        admin_h = {"Authorization": f"Bearer {admin_creds[2]}"}
        gid = f"e2e_grp_{uuid.uuid4().hex[:6]}"
        mid = f"e2e_mlp_{uuid.uuid4().hex[:6]}"
        _seed_group_with_model(fresh_client, admin_h, gid, mid, n_params=73)

        # Force 4 chunks
        ckpt_path = Path("models") / "global" / gid / "model_latest.pt"
        total = ckpt_path.stat().st_size
        chunk_size = total // 4

        manifest = fresh_client.post(
            "/api/downloads/init",
            json={"group_id": gid, "format": "pt", "chunk_size": chunk_size},
            headers=admin_h,
        ).json()

        parts = []
        for i, chunk_meta in enumerate(manifest["chunks"]):
            r = fresh_client.get(chunk_meta["url"])
            assert r.status_code == 200
            assert r.headers["X-Chunk-Index"] == str(i)
            parts.append(r.content)

        assembled = b"".join(parts)
        assert hashlib.sha256(assembled).hexdigest() == manifest["sha256"]

        comp = fresh_client.post(
            f"/api/downloads/{manifest['download_id']}/complete",
            headers=admin_h,
        ).json()
        assert comp["status"] == "completed"
        assert comp["bytes_served"] == total

    def test_large_file_chunked_end_to_end(
        self, fresh_client, admin_creds, temp_models_dir
    ):
        """Write a 10 MB synthetic .pt and verify chunking works at scale."""
        admin_h = {"Authorization": f"Bearer {admin_creds[2]}"}
        gid = f"big_grp_{uuid.uuid4().hex[:6]}"
        mid = f"big_mlp_{uuid.uuid4().hex[:6]}"

        # 10 MB of random bytes is enough params for any reasonable model
        big_size = 10 * 1024 * 1024
        n_params = big_size // 4
        _seed_group_with_model(fresh_client, admin_h, gid, mid, n_params=n_params)

        # Use default 8 MB chunks → should be 2 chunks
        manifest = fresh_client.post(
            "/api/downloads/init",
            json={"group_id": gid, "format": "pt"},
            headers=admin_h,
        ).json()
        assert manifest["num_chunks"] == 2

        parts = []
        for chunk_meta in manifest["chunks"]:
            r = fresh_client.get(chunk_meta["url"])
            assert r.status_code == 200
            parts.append(r.content)
        assert sum(len(p) for p in parts) == manifest["total_size"]
        reassembled = b"".join(parts)
        assert hashlib.sha256(reassembled).hexdigest() == manifest["sha256"]
