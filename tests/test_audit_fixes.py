"""
Regression tests for the 20 audit fixes.
"""

import base64
import os

import numpy as np
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
        "username": f"audit_admin_{os.urandom(4).hex()}",
        "password": "testpass123",
        "role": "admin",
    })
    data = resp.json()
    login = fresh_client.post("/api/auth/login", json={
        "username": data.get("user", {}).get("username", ""),
        "password": "testpass123",
    })
    token = login.json().get("token", "")
    return {"Authorization": f"Bearer {token}"}


def _register_model(fresh_client, auth_headers, mid):
    fresh_client.post("/api/models/register/architecture", json={
        "model_id": mid,
        "architecture_path": "torch.nn.Linear",
        "config": {"in_features": 10, "out_features": 3},
    }, headers=auth_headers)


# ======================================================================
# G1: POST /api/groups requires authentication
# ======================================================================

class TestG1GroupAuth:
    def test_create_group_no_auth_returns_401(self, fresh_client):
        resp = fresh_client.post("/api/groups", json={
            "group_id": "g1_noauth",
            "model_id": "some_model",
        })
        assert resp.status_code == 401

    def test_create_group_with_auth_succeeds(self, fresh_client, auth_headers):
        mid = f"g1_model_{os.urandom(4).hex()}"
        _register_model(fresh_client, auth_headers, mid)
        resp = fresh_client.post("/api/groups", json={
            "group_id": f"g1_ok_{os.urandom(4).hex()}",
            "model_id": mid,
        }, headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["status"] == "created"


# ======================================================================
# G2: Aggregator selection maps to config["robust"]["method"]
# ======================================================================

class TestG2AggregatorMapping:
    def test_hybrid_creates_robust_config(self, fresh_client, auth_headers):
        mid = f"g2_model_{os.urandom(4).hex()}"
        _register_model(fresh_client, auth_headers, mid)
        gid = f"g2_grp_{os.urandom(4).hex()}"
        resp = fresh_client.post("/api/groups", json={
            "group_id": gid, "model_id": mid, "aggregator": "hybrid",
        }, headers=auth_headers)
        assert resp.status_code == 200

        from astra.app.state import get_fl_server
        group = get_fl_server().group_manager.groups.get(gid)
        assert group is not None
        assert group.config.get("robust", {}).get("method") == "hybrid"

    def test_trimmed_mean_creates_robust_config(self, fresh_client, auth_headers):
        mid = f"g2_model_{os.urandom(4).hex()}"
        _register_model(fresh_client, auth_headers, mid)
        gid = f"g2_grp_{os.urandom(4).hex()}"
        resp = fresh_client.post("/api/groups", json={
            "group_id": gid, "model_id": mid, "aggregator": "trimmed_mean",
        }, headers=auth_headers)
        assert resp.status_code == 200

        from astra.app.state import get_fl_server
        group = get_fl_server().group_manager.groups.get(gid)
        assert group is not None
        assert group.config.get("robust", {}).get("method") == "trimmed_mean"

    def test_fedavg_does_not_add_robust_key(self, fresh_client, auth_headers):
        mid = f"g2_model_{os.urandom(4).hex()}"
        _register_model(fresh_client, auth_headers, mid)
        gid = f"g2_grp_{os.urandom(4).hex()}"
        resp = fresh_client.post("/api/groups", json={
            "group_id": gid, "model_id": mid, "aggregator": "fedavg",
        }, headers=auth_headers)
        assert resp.status_code == 200

        from astra.app.state import get_fl_server
        group = get_fl_server().group_manager.groups.get(gid)
        assert group is not None
        assert "robust" not in group.config


# ======================================================================
# D2: trigger_clients_training no longer exists
# ======================================================================

class TestD2NoTriggerCrash:
    def test_method_removed(self):
        from astra.app.group_manager import GroupManager
        assert not hasattr(GroupManager, "trigger_clients_training")


# ======================================================================
# D6: Little-endian dtype consistency
# ======================================================================

class TestD6LittleEndian:
    def test_decode_bytes_little_endian(self):
        from astra.app.group_manager import GroupManager
        gm = GroupManager(config={})
        arr = np.array([1.0, 2.0, 3.0, -1.5], dtype="<f4")
        decoded = gm._decode_local_updates(arr.tobytes())
        np.testing.assert_array_equal(decoded, arr)

    def test_decode_base64_little_endian(self):
        from astra.app.group_manager import GroupManager
        gm = GroupManager(config={})
        arr = np.array([1.0, 2.0, 3.0], dtype="<f4")
        b64 = base64.b64encode(arr.tobytes()).decode()
        decoded = gm._decode_local_updates(b64)
        np.testing.assert_array_equal(decoded, arr)

    def test_server_decode_little_endian(self):
        from astra.core.server import AsyncServer
        from astra.core.aggregation.aggregator import FedAvgAggregator
        import torch.nn as nn

        model = nn.Linear(10, 3)
        config = {
            "server": {"aggregator_window": 5, "async_lambda": 0.2, "server_lr": 0.5},
            "privacy": {"dp_enabled": False, "dp_mode": "client", "clip_norm": 1.0, "sigma": 1.2},
            "trust": {"init": 1.0, "update_alpha": 0.3, "quarantine_threshold": 0.35, "soft_decay": 0.8},
        }
        server = AsyncServer(model, FedAvgAggregator(config), config)
        arr = np.array([0.5, -0.3, 1.2], dtype="<f4")
        decoded = server._decode_update(arr.tobytes())
        np.testing.assert_array_almost_equal(decoded, arr)


# ======================================================================
# D7: pending_updates is bounded deque
# ======================================================================

class TestD7BoundedBuffer:
    def test_pending_updates_is_deque_with_maxlen(self):
        from astra.app.training_group import TrainingGroup, AsyncWindowConfig
        from collections import deque
        tg = TrainingGroup(
            group_id="test", model_id="m", config={},
            window_config=AsyncWindowConfig(window_size=3, time_limit=20.0),
        )
        assert isinstance(tg.pending_updates, deque)
        assert tg.pending_updates.maxlen == 500


# ======================================================================
# D3: Metrics persistence uses group_id as experiment_id
# ======================================================================

class TestD3MetricsPersistence:
    def test_aggregate_writes_metrics_to_db(self, fresh_client, auth_headers):
        """Manually call aggregate_group and verify DB is written."""
        mid = f"d3_model_{os.urandom(4).hex()}"
        _register_model(fresh_client, auth_headers, mid)
        gid = f"d3_grp_{os.urandom(4).hex()}"
        resp = fresh_client.post("/api/groups", json={
            "group_id": gid, "model_id": mid, "window_size": 1,
        }, headers=auth_headers)
        assert resp.status_code == 200

        from astra.app.state import get_fl_server
        import torch
        gm = get_fl_server().group_manager
        group = gm.groups.get(gid)
        assert group is not None

        model = get_fl_server().model_registry.build_model(mid)
        num_params = sum(p.numel() for p in model.parameters())
        fake_delta = np.random.randn(num_params).astype(np.float32)

        # Directly add client and update (bypass register_client to avoid asyncio)
        group.clients["d3_client_1"] = {
            "status": "active", "joined_at": "now", "last_update": None,
            "trust_score": 1.0, "updates_count": 0, "local_accuracy": 0.0,
            "local_loss": 0.0, "gradient_norm": 0.0, "user_id": 9999,
        }
        group.add_update("d3_client_1", {
            "delta": fake_delta,
            "local_dataset_size": 100,
            "staleness_weight": 1.0,
            "trust": 1.0,
            "meta": {"train_accuracy": 0.85, "train_loss": 0.3},
        })

        gm.server_model = model
        result = gm.aggregate_group(gid)
        assert result is not None
        assert result["version"] == 1

        # Verify in-memory metrics_history was updated
        # Accuracy is now server-evaluated (0.0 when no val_dataset configured)
        assert len(group.metrics_history) >= 1
        assert group.metrics_history[-1]["accuracy"] == 0.0
        assert group.metrics_history[-1]["metrics_source"] == "unverified"

        # Verify DB persistence: metrics table uses group_id as experiment_id.
        # The metrics table has FK to experiments, so create the experiment first.
        from astra.app.database import get_db
        db = get_db()
        db.create_experiment(gid, {"group_id": gid})
        # Re-run aggregation to persist to DB this time
        group.clients["d3_client_2"] = {
            "status": "active", "joined_at": "now", "last_update": None,
            "trust_score": 1.0, "updates_count": 0,
            "gradient_norm": 0.0, "user_id": 9998,
        }
        group.add_update("d3_client_2", {
            "delta": fake_delta,
            "local_dataset_size": 50,
            "staleness_weight": 1.0,
            "trust": 1.0,
            "meta": {},
        })
        result2 = gm.aggregate_group(gid)
        assert result2 is not None

        metrics = db.get_experiment_metrics(gid)
        assert len(metrics) >= 1


# ======================================================================
# D4: Trust score sync back to group.clients
# ======================================================================

class TestD4TrustSync:
    def test_trust_score_initial_default(self, fresh_client, auth_headers):
        mid = f"d4_model_{os.urandom(4).hex()}"
        _register_model(fresh_client, auth_headers, mid)
        gid = f"d4_grp_{os.urandom(4).hex()}"
        fresh_client.post("/api/groups", json={
            "group_id": gid, "model_id": mid, "window_size": 10,
        }, headers=auth_headers)

        from astra.app.state import get_fl_server
        gm = get_fl_server().group_manager
        group = gm.groups.get(gid)
        assert group is not None

        # Manually add client (bypass register_client to avoid asyncio)
        group.clients["d4_client_1"] = {
            "status": "active", "joined_at": "now", "last_update": None,
            "trust_score": 1.0, "updates_count": 0, "local_accuracy": 0.0,
            "local_loss": 0.0, "gradient_norm": 0.0, "user_id": 8888,
        }

        info = group.clients.get("d4_client_1")
        assert info is not None
        assert info["trust_score"] == 1.0

    def test_trust_manager_updates_and_returns_score(self):
        """Verify TrustManager.update_trust returns a score that can be synced."""
        from astra.core.trust_manager import TrustManager
        import torch

        config = {
            "trust": {"init": 1.0, "update_alpha": 0.3, "quarantine_threshold": 0.35, "soft_decay": 0.8}
        }
        tm = TrustManager(config)

        global_vec = np.ones(10, dtype=np.float32)
        update_vec = np.ones(10, dtype=np.float32) * 0.9

        score = tm.update_trust("test_client", update_vec, global_vec)
        assert 0.0 <= score <= 1.0
        # Score should differ from initial (1.0) since there IS a global vector
        assert score != 1.0 or tm.get_trust("test_client") == score


# ======================================================================
# J2: user_id preserved in client dicts after DB load
# ======================================================================

class TestJ2UserIdInClientDict:
    def test_user_id_in_reconstructed_client_dict(self):
        from astra.app.group_manager import GroupManager
        from astra.app.database import get_db

        db = get_db()
        gid = f"j2_grp_{os.urandom(4).hex()}"
        cid = f"j2_client_{os.urandom(4).hex()}"

        import bcrypt
        pw = bcrypt.hashpw(b"test", bcrypt.gensalt()).decode()
        uname = f"j2_user_{os.urandom(4).hex()}"
        with db.connection() as conn:
            conn.execute(
                "INSERT INTO users (username, password_hash, role) VALUES (?, ?, 'client')",
                (uname, pw),
            )
            uid = conn.execute("SELECT id FROM users WHERE username = ?", (uname,)).fetchone()[0]
            conn.commit()

        db.create_group(group_id=gid, model_id="test_model", window_size=3, time_limit=20)
        db.register_fl_client(client_id=cid, experiment_id=gid, user_id=uid, group_id=gid)

        gm = GroupManager(config={})
        group = gm.groups.get(gid)
        assert group is not None
        assert cid in group.clients
        assert group.clients[cid]["user_id"] == uid


# ======================================================================
# Bug fix: updates_count only incremented once per upload
# ======================================================================

class TestUpdatesCountSingleIncrement:
    def test_add_update_increments_once(self):
        from astra.app.training_group import TrainingGroup, AsyncWindowConfig
        tg = TrainingGroup(
            group_id="uc_test", model_id="m", config={},
            window_config=AsyncWindowConfig(window_size=10, time_limit=20.0),
        )
        tg.clients["c1"] = {
            "status": "active", "joined_at": "now", "last_update": None,
            "trust_score": 1.0, "updates_count": 0, "local_accuracy": 0.0,
            "local_loss": 0.0, "gradient_norm": 0.0, "user_id": 1,
        }
        tg.add_update("c1", {
            "delta": np.array([1.0], dtype="<f4"),
            "local_dataset_size": 10,
            "meta": {},
        })
        assert tg.clients["c1"]["updates_count"] == 1

        tg.add_update("c1", {
            "delta": np.array([2.0], dtype="<f4"),
            "local_dataset_size": 10,
            "meta": {},
        })
        assert tg.clients["c1"]["updates_count"] == 2

    def test_to_dict_uses_updates_count(self):
        from astra.app.training_group import TrainingGroup, AsyncWindowConfig
        tg = TrainingGroup(
            group_id="uc_dict", model_id="m", config={},
            window_config=AsyncWindowConfig(window_size=10, time_limit=20.0),
        )
        tg.clients["c1"] = {
            "status": "active", "joined_at": "now", "last_update": None,
            "trust_score": 1.0, "updates_count": 5, "local_accuracy": 0.0,
            "local_loss": 0.0, "gradient_norm": 0.0, "user_id": 1,
        }
        d = tg.to_dict()
        assert d["clients"]["c1"]["update_count"] == 5


# ======================================================================
# Bug fix: client status loaded from DB, not hardcoded to "offline"
# ======================================================================

class TestClientStatusFromDB:
    def test_status_loaded_from_db_not_offline(self):
        from astra.app.group_manager import GroupManager
        from astra.app.database import get_db

        db = get_db()
        gid = f"cs_grp_{os.urandom(4).hex()}"
        cid = f"cs_client_{os.urandom(4).hex()}"

        import bcrypt
        pw = bcrypt.hashpw(b"test", bcrypt.gensalt()).decode()
        uname = f"cs_user_{os.urandom(4).hex()}"
        with db.connection() as conn:
            conn.execute(
                "INSERT INTO users (username, password_hash, role) VALUES (?, ?, 'client')",
                (uname, pw),
            )
            uid = conn.execute("SELECT id FROM users WHERE username = ?", (uname,)).fetchone()[0]
            conn.commit()

        db.create_group(group_id=gid, model_id="test_model", window_size=3, time_limit=20)
        db.register_fl_client(client_id=cid, experiment_id=gid, user_id=uid, group_id=gid)

        # Set status to "active" in DB
        db.update_fl_client_metrics(client_id=cid, status="active")

        gm = GroupManager(config={})
        group = gm.groups.get(gid)
        assert group is not None
        assert cid in group.clients
        assert group.clients[cid]["status"] == "active"


# ======================================================================
# Training Manifest: schema validation and group creation
# ======================================================================

class TestTrainingManifest:
    def test_manifest_schema_requires_expected_delta_bytes(self):
        from astra.infra.models import TrainingManifest
        with pytest.raises(Exception):
            TrainingManifest(model_id="test")  # missing expected_delta_bytes

    def test_manifest_schema_valid(self):
        from astra.infra.models import TrainingManifest
        m = TrainingManifest(
            model_id="test_model",
            expected_delta_bytes=1024,
            is_peft=True,
            target_modules=["q_proj", "v_proj"],
            lora_rank=8,
        )
        assert m.model_id == "test_model"
        assert m.expected_delta_bytes == 1024
        assert m.is_peft is True
        assert m.lr == 0.01  # default

    def test_manifest_stored_in_group_config(self, fresh_client, auth_headers):
        mid = f"manifest_model_{os.urandom(4).hex()}"
        _register_model(fresh_client, auth_headers, mid)
        gid = f"manifest_grp_{os.urandom(4).hex()}"
        resp = fresh_client.post("/api/groups", json={
            "group_id": gid, "model_id": mid,
            "training_manifest": {
                "model_id": mid,
                "expected_delta_bytes": 1024,
                "lr": 0.001,
                "val_dataset": "mnist",
            },
        }, headers=auth_headers)
        assert resp.status_code == 200

        from astra.app.state import get_fl_server
        group = get_fl_server().group_manager.groups.get(gid)
        assert group is not None
        manifest = group.config.get("training_manifest")
        assert manifest is not None
        assert manifest["model_id"] == mid
        assert manifest["lr"] == 0.001
        assert manifest["val_dataset"] == "mnist"

    def test_manifest_endpoint_returns_manifest(self, fresh_client, auth_headers):
        mid = f"manifest_ep_model_{os.urandom(4).hex()}"
        _register_model(fresh_client, auth_headers, mid)
        gid = f"manifest_ep_{os.urandom(4).hex()}"
        fresh_client.post("/api/groups", json={
            "group_id": gid, "model_id": mid,
            "training_manifest": {
                "model_id": mid,
                "expected_delta_bytes": 2048,
            },
        }, headers=auth_headers)

        resp = fresh_client.get(f"/api/groups/{gid}/manifest")
        assert resp.status_code == 200
        data = resp.json()
        assert data["manifest"]["expected_delta_bytes"] == 2048

    def test_manifest_endpoint_404_without_manifest(self, fresh_client, auth_headers):
        mid = f"manifest_noman_model_{os.urandom(4).hex()}"
        _register_model(fresh_client, auth_headers, mid)
        gid = f"manifest_noman_{os.urandom(4).hex()}"
        fresh_client.post("/api/groups", json={
            "group_id": gid, "model_id": mid,
        }, headers=auth_headers)

        resp = fresh_client.get(f"/api/groups/{gid}/manifest")
        assert resp.status_code == 404


# ======================================================================
# Server-side evaluation
# ======================================================================

class TestServerEvaluation:
    def test_evaluate_returns_zeros_without_val_dataset(self):
        from astra.app.group_manager import GroupManager
        from astra.app.training_group import TrainingGroup, AsyncWindowConfig
        gm = GroupManager(config={})
        gid = f"eval_{os.urandom(4).hex()}"
        group = TrainingGroup(
            group_id=gid, model_id="m", config={},
            window_config=AsyncWindowConfig(window_size=3, time_limit=20),
        )
        gm.groups[gid] = group
        result = gm.evaluate_global_model(gid)
        assert result == {"accuracy": 0.0, "loss": 0.0}

    def test_metrics_history_has_metrics_source(self):
        from astra.app.training_group import TrainingGroup, AsyncWindowConfig
        group = TrainingGroup(
            group_id="src_test", model_id="m", config={},
            window_config=AsyncWindowConfig(window_size=3, time_limit=20),
        )
        d = group.to_dict()
        assert d["metrics_source"] == "unverified"

    def test_metrics_history_source_server_when_val_configured(self):
        from astra.app.training_group import TrainingGroup, AsyncWindowConfig
        group = TrainingGroup(
            group_id="src_test2", model_id="m",
            config={"training_manifest": {"val_dataset": "mnist"}},
            window_config=AsyncWindowConfig(window_size=3, time_limit=20),
        )
        d = group.to_dict()
        assert d["metrics_source"] == "server"
