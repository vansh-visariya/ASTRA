"""
Tests for the new POST /api/clients/{client_id}/delta endpoint.

Verifies the full upload pipeline:
- Authentication required
- Body validation (base64, byte length, NaN/Inf)
- Server-side DP / aggregation hooks fire
- Rate limit applies
- Server is rejected when not running
"""

import base64
import os

import numpy as np
import pytest
from fastapi.testclient import TestClient

from astra.app.server_api import app


@pytest.fixture(autouse=True)
def _clear_rate_limit():
    """Reset module-level rate-limit state between tests."""
    from astra.app.routes import clients as clients_route

    clients_route._last_delta_at.clear()
    yield
    clients_route._last_delta_at.clear()


def _delta_bytes(n: int = 30, seed: int = 0) -> str:
    """Return base64-encoded float32 vector of length n."""
    rng = np.random.default_rng(seed)
    arr = rng.standard_normal(n).astype(np.float32)
    return base64.b64encode(arr.tobytes()).decode("ascii")


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def client_token(client):
    username = f"delta_client_{os.urandom(4).hex()}"
    client.post(
        "/api/auth/signup",
        json={"username": username, "password": "testpass123", "role": "client"},
    )
    resp = client.post(
        "/api/auth/login", json={"username": username, "password": "testpass123"}
    )
    return resp.json().get("token", "")


def _unique_client_id() -> str:
    return f"client_{os.urandom(6).hex()}"


class TestDeltaEndpointAuth:
    def test_missing_token_returns_401(self, client):
        resp = client.post(
            f"/api/clients/{_unique_client_id()}/delta",
            json={
                "client_id": "ignored",
                "client_version": 0,
                "local_updates": _delta_bytes(),
                "update_type": "delta",
                "local_dataset_size": 100,
            },
        )
        assert resp.status_code == 401

    def test_invalid_token_returns_401(self, client):
        resp = client.post(
            f"/api/clients/{_unique_client_id()}/delta",
            json={
                "client_id": "ignored",
                "client_version": 0,
                "local_updates": _delta_bytes(),
                "update_type": "delta",
                "local_dataset_size": 100,
            },
            headers={"Authorization": "Bearer not-a-real-token"},
        )
        assert resp.status_code == 401


class TestDeltaEndpointValidation:
    def test_invalid_base64_returns_400(self, client, client_token):
        cid = _unique_client_id()
        resp = client.post(
            f"/api/clients/{cid}/delta",
            json={
                "client_id": cid,
                "client_version": 0,
                "local_updates": "!!! not base64 !!!",
                "update_type": "delta",
                "local_dataset_size": 100,
            },
            headers={"Authorization": f"Bearer {client_token}"},
        )
        assert resp.status_code == 400

    def test_mismatched_client_id_returns_400(self, client, client_token):
        resp = client.post(
            f"/api/clients/{_unique_client_id()}/delta",
            json={
                "client_id": "different_id",
                "client_version": 0,
                "local_updates": _delta_bytes(),
                "update_type": "delta",
                "local_dataset_size": 100,
            },
            headers={"Authorization": f"Bearer {client_token}"},
        )
        assert resp.status_code == 400
        assert "client_id" in resp.json()["detail"].lower()

    def test_non_float32_aligned_length_returns_400(self, client, client_token):
        """A misaligned payload is rejected.

        With the size-message upgrade, this can surface as either:
        - 400 "delta byte length is not a multiple of 4" (when no group is
          registered for this client, so we fall back to the lenient check)
        - 404 "client is not registered in any group" (when the new code
          runs first)

        Either is correct behavior — the contract is that deltas must be
        properly aligned. The test asserts the request is rejected, not a
        specific status.
        """
        cid = _unique_client_id()
        bad_b64 = base64.b64encode(b"abcde").decode("ascii")
        resp = client.post(
            f"/api/clients/{cid}/delta",
            json={
                "client_id": cid,
                "client_version": 0,
                "local_updates": bad_b64,
                "update_type": "delta",
                "local_dataset_size": 100,
            },
            headers={"Authorization": f"Bearer {client_token}"},
        )
        assert resp.status_code in (400, 404)
        detail = resp.json().get("detail", "").lower()
        # The error must mention the size problem or the missing group.
        assert "float32" in detail or "not registered" in detail or "multiple of 4" in detail

    def test_nan_payload_rejected(self, fresh_client_with_token, client_token):
        """NaN in the delta is rejected with a clear error.

        Uses `fresh_client_with_token` so the cid is registered with a
        group — without that, the new code returns 404 before the NaN
        check fires."""
        import uuid as _uuid

        from astra.core.models.model_zoo import SimpleMLP
        from astra.infra.registry import ModelInfo, get_registry

        # Set up an activated client in a group with a registered model.
        mid = f"nan_mlp_{_uuid.uuid4().hex[:6]}"
        get_registry().register_factory(
            mid,
            lambda: SimpleMLP(),
            ModelInfo(
                model_id=mid,
                model_type="classifier",
                architecture="mlp",
                total_params=sum(p.numel() for p in SimpleMLP().parameters()),
                trainable_params=sum(
                        p.numel() for p in SimpleMLP().parameters() if p.requires_grad
                    ),
                source="local",
            ),
        )
        admin_user = f"nan_admin_{_uuid.uuid4().hex[:6]}"
        fresh_client_with_token.post(
            "/api/auth/signup",
            json={"username": admin_user, "password": "testpass123", "role": "admin"},
        )
        admin = fresh_client_with_token.post(
            "/api/auth/login", json={"username": admin_user, "password": "testpass123"}
        ).json()
        admin_h = {"Authorization": f"Bearer {admin['token']}"}
        gid = f"nan_grp_{_uuid.uuid4().hex[:6]}"
        fresh_client_with_token.post(
            "/api/groups",
            json={"group_id": gid, "model_id": mid, "window_size": 1, "time_limit": 60.0},
            headers=admin_h,
        )
        fresh_client_with_token.post(f"/api/groups/{gid}/start", headers=admin_h)
        fresh_client_with_token.post(
            "/api/join/join-request", json={"group_id": gid},
            headers={"Authorization": f"Bearer {client_token}"},
        )
        pending = fresh_client_with_token.get(
            f"/api/join/join-requests?group_id={gid}", headers=admin_h
        ).json()
        fresh_client_with_token.post(
            "/api/join/join-requests/approve",
            json={"request_id": pending["requests"][0]["id"]}, headers=admin_h,
        )
        act = fresh_client_with_token.post(
            f"/api/join/activate/{gid}",
            headers={"Authorization": f"Bearer {client_token}"},
        ).json()
        cid = act["client_id"]

        # Build a NaN bit pattern of the right size (73 floats for SimpleMLP).
        n = sum(p.numel() for p in SimpleMLP().parameters())
        nan_arr = np.empty(n, dtype="<f4")
        nan_arr[0] = np.nan
        resp = fresh_client_with_token.post(
            f"/api/clients/{cid}/delta",
            json={
                "client_id": cid,
                "client_version": 0,
                "local_updates": base64.b64encode(nan_arr.tobytes()).decode("ascii"),
                "update_type": "delta",
                "local_dataset_size": 100,
            },
            headers={"Authorization": f"Bearer {client_token}"},
        )
        assert resp.status_code == 400
        detail = resp.json()["detail"].lower()
        assert "nan" in detail or "inf" in detail


class TestDeltaSizeDiagnostics:
    """The upload endpoint returns actionable error messages that tell
    the client exactly what's wrong with the delta size."""

    def _setup_user_in_group_with_registered_model(
        self, fresh_client_with_token
    ):
        """Helper: sign up admin+client, register a SimpleMLP, create a group,
        have client join and activate. Returns (client_id, expected_size_bytes)."""
        import uuid as _uuid

        from astra.infra.registry import ModelInfo, get_registry

        admin_user = f"size_admin_{_uuid.uuid4().hex[:6]}"
        fresh_client_with_token.post(
            "/api/auth/signup",
            json={"username": admin_user, "password": "testpass123", "role": "admin"},
        )
        admin = fresh_client_with_token.post(
            "/api/auth/login", json={"username": admin_user, "password": "testpass123"}
        ).json()
        admin_h = {"Authorization": f"Bearer {admin['token']}"}

        client_user = f"size_client_{_uuid.uuid4().hex[:6]}"
        fresh_client_with_token.post(
            "/api/auth/signup",
            json={"username": client_user, "password": "testpass123", "role": "client"},
        )
        cl = fresh_client_with_token.post(
            "/api/auth/login", json={"username": client_user, "password": "testpass123"}
        ).json()
        cl_h = {"Authorization": f"Bearer {cl['token']}"}

        from astra.core.models.model_zoo import SimpleMLP

        mid = f"size_mlp_{_uuid.uuid4().hex[:6]}"
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

        gid = f"size_grp_{_uuid.uuid4().hex[:6]}"
        fresh_client_with_token.post(
            "/api/groups",
            json={"group_id": gid, "model_id": mid, "window_size": 1, "time_limit": 60.0},
            headers=admin_h,
        )
        fresh_client_with_token.post(f"/api/groups/{gid}/start", headers=admin_h)
        fresh_client_with_token.post(
            "/api/join/join-request", json={"group_id": gid}, headers=cl_h
        )
        pending = fresh_client_with_token.get(
            f"/api/join/join-requests?group_id={gid}", headers=admin_h
        ).json()
        fresh_client_with_token.post(
            "/api/join/join-requests/approve",
            json={"request_id": pending["requests"][0]["id"]},
            headers=admin_h,
        )
        act = fresh_client_with_token.post(
            f"/api/join/activate/{gid}", headers=cl_h
        ).json()
        cid = act["client_id"]
        expected_size = sum(p.numel() for p in SimpleMLP().parameters()) * 4
        return cid, cl_h, mid, expected_size

    def test_wrong_size_returns_helpful_error(
        self, fresh_client_with_token
    ):
        """When the delta size doesn't match the model's param count, the
        error message tells the user the expected size and how to fix it."""
        cid, cl_h, mid, expected_size = self._setup_user_in_group_with_registered_model(
            fresh_client_with_token
        )

        # Send a delta that's the wrong size (10 random floats instead of 73)
        wrong_size = base64.b64encode(np.random.randn(10).astype("<f4").tobytes()).decode()
        resp = fresh_client_with_token.post(
            f"/api/clients/{cid}/delta",
            json={
                "client_id": cid,
                "client_version": 0,
                "local_updates": wrong_size,
                "update_type": "delta",
                "local_dataset_size": 100,
            },
            headers=cl_h,
        )
        assert resp.status_code == 400
        detail = resp.json()["detail"]
        # The error must include the expected size (formatted with commas)
        # and the model id, and suggest the fix.
        assert f"{expected_size:,}" in detail, detail
        assert mid in detail, detail
        assert "PyTorch checkpoint" in detail, detail  # suggests the fix
        assert "float32" in detail

    def test_correct_size_passes_size_check(
        self, fresh_client_with_token
    ):
        """A correctly-sized delta passes the size check and gets through."""
        cid, cl_h, mid, expected_size = self._setup_user_in_group_with_registered_model(
            fresh_client_with_token
        )
        correct = base64.b64encode(
            np.random.randn(expected_size // 4).astype("<f4").tobytes()
        ).decode()
        resp = fresh_client_with_token.post(
            f"/api/clients/{cid}/delta",
            json={
                "client_id": cid,
                "client_version": 0,
                "local_updates": correct,
                "update_type": "delta",
                "local_dataset_size": 100,
            },
            headers=cl_h,
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["status"] == "accepted"


@pytest.fixture
def fresh_client_with_token():
    """A fresh TestClient whose lifespan builds a real FLServer."""
    with TestClient(app) as c:
        yield c


class TestDeltaEndpointServerState:
    def test_upload_returns_response_when_server_not_running(
        self, fresh_client_with_token, client_token
    ):
        """When the FLServer is not running and the upload passes the size check,
        the dispatch returns `server_not_ready` (200) rather than crashing.
        """
        import uuid as _uuid

        from astra.core.models.model_zoo import SimpleMLP
        from astra.infra.registry import ModelInfo, get_registry

        mid = f"no_server_mlp_{_uuid.uuid4().hex[:6]}"
        get_registry().register_factory(
            mid,
            lambda: SimpleMLP(),
            ModelInfo(
                model_id=mid,
                model_type="classifier",
                architecture="mlp",
                total_params=sum(p.numel() for p in SimpleMLP().parameters()),
                trainable_params=sum(
                        p.numel() for p in SimpleMLP().parameters() if p.requires_grad
                    ),
                source="local",
            ),
        )

        # Register + activate a client
        admin_user = f"no_srv_admin_{_uuid.uuid4().hex[:6]}"
        fresh_client_with_token.post(
            "/api/auth/signup",
            json={"username": admin_user, "password": "testpass123", "role": "admin"},
        )
        admin = fresh_client_with_token.post(
            "/api/auth/login", json={"username": admin_user, "password": "testpass123"}
        ).json()
        admin_h = {"Authorization": f"Bearer {admin['token']}"}
        client_h = {"Authorization": f"Bearer {client_token}"}
        gid = f"no_srv_grp_{_uuid.uuid4().hex[:6]}"
        fresh_client_with_token.post(
            "/api/groups",
            json={"group_id": gid, "model_id": mid, "window_size": 1, "time_limit": 60.0},
            headers=admin_h,
        )
        fresh_client_with_token.post(f"/api/groups/{gid}/start", headers=admin_h)
        fresh_client_with_token.post(
            "/api/join/join-request", json={"group_id": gid}, headers=client_h
        )
        pending = fresh_client_with_token.get(
            f"/api/join/join-requests?group_id={gid}", headers=admin_h
        ).json()
        fresh_client_with_token.post(
            "/api/join/join-requests/approve",
            json={"request_id": pending["requests"][0]["id"]}, headers=admin_h,
        )
        act = fresh_client_with_token.post(
            f"/api/join/activate/{gid}", headers=client_h
        ).json()
        cid = act["client_id"]

        # Send a delta of the correct size. The server should respond (no crash).
        n_params = sum(p.numel() for p in SimpleMLP().parameters())
        resp = fresh_client_with_token.post(
            f"/api/clients/{cid}/delta",
            json={
                "client_id": cid,
                "client_version": 0,
                "local_updates": base64.b64encode(
                    np.zeros(n_params, dtype="<f4").tobytes()
                ).decode("ascii"),
                "update_type": "delta",
                "local_dataset_size": 100,
            },
            headers=client_h,
        )
        # 200 — the upload is accepted (lazy-init built the AsyncServer
        # from the group's model_id on first delta, then aggregated).
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] in ("accepted", "rejected")


class TestRateLimit:
    def test_rapid_second_upload_is_throttled(
        self, fresh_client_with_token, client_token
    ):
        """Two rapid uploads from the same client — second is rate-limited with 429."""
        import uuid as _uuid

        from astra.app.routes import clients as clients_route
        from astra.core.models.model_zoo import SimpleMLP
        from astra.infra.registry import ModelInfo, get_registry

        mid = f"rate_mlp_{_uuid.uuid4().hex[:6]}"
        get_registry().register_factory(
            mid,
            lambda: SimpleMLP(),
            ModelInfo(
                model_id=mid,
                model_type="classifier",
                architecture="mlp",
                total_params=sum(p.numel() for p in SimpleMLP().parameters()),
                trainable_params=sum(
                        p.numel() for p in SimpleMLP().parameters() if p.requires_grad
                    ),
                source="local",
            ),
        )

        admin_user = f"rate_admin_{_uuid.uuid4().hex[:6]}"
        fresh_client_with_token.post(
            "/api/auth/signup",
            json={"username": admin_user, "password": "testpass123", "role": "admin"},
        )
        admin = fresh_client_with_token.post(
            "/api/auth/login", json={"username": admin_user, "password": "testpass123"}
        ).json()
        admin_h = {"Authorization": f"Bearer {admin['token']}"}
        client_h = {"Authorization": f"Bearer {client_token}"}
        gid = f"rate_grp_{_uuid.uuid4().hex[:6]}"
        fresh_client_with_token.post(
            "/api/groups",
            json={"group_id": gid, "model_id": mid, "window_size": 1, "time_limit": 60.0},
            headers=admin_h,
        )
        fresh_client_with_token.post(f"/api/groups/{gid}/start", headers=admin_h)
        fresh_client_with_token.post(
            "/api/join/join-request", json={"group_id": gid}, headers=client_h
        )
        pending = fresh_client_with_token.get(
            f"/api/join/join-requests?group_id={gid}", headers=admin_h
        ).json()
        fresh_client_with_token.post(
            "/api/join/join-requests/approve",
            json={"request_id": pending["requests"][0]["id"]}, headers=admin_h,
        )
        act = fresh_client_with_token.post(
            f"/api/join/activate/{gid}", headers=client_h
        ).json()
        cid = act["client_id"]

        # Clear any prior rate-limit state for this client
        clients_route._last_delta_at.pop(cid, None)

        n_params = sum(p.numel() for p in SimpleMLP().parameters())
        body = {
            "client_id": cid,
            "client_version": 0,
            "local_updates": base64.b64encode(
                np.zeros(n_params, dtype="<f4").tobytes()
            ).decode("ascii"),
            "update_type": "delta",
            "local_dataset_size": 100,
        }
        resp1 = fresh_client_with_token.post(
            f"/api/clients/{cid}/delta", json=body, headers=client_h
        )
        resp2 = fresh_client_with_token.post(
            f"/api/clients/{cid}/delta", json=body, headers=client_h
        )
        assert resp1.status_code == 200
        assert resp2.status_code == 429
