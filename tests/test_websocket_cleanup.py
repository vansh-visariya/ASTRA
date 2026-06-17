"""
Tests verifying the WebSocket handler rejects client-training messages.

After removing the in-process FLClient, the WebSocket is no longer a
channel for client training. These tests run the websocket_endpoint
function directly against a fake websocket — no TestClient / no
lifespan — so they're fast and isolated.
"""

import asyncio
import contextlib
import json

import pytest

from astra.infra.websocket_handler import websocket_endpoint


class FakeWebSocket:
    """Minimal stand-in for fastapi.WebSocket that the handler can use."""

    def __init__(self, token=None, send_closed=True):
        self.sent: list[dict] = []
        self.received: list[str] = []
        self.accepted = False
        self.closed_code = None
        self.query_params = {"token": token} if token else {}

    async def accept(self):
        self.accepted = True

    async def close(self, code=1000):
        self.closed_code = code

    async def send_json(self, data):
        self.sent.append(data)

    async def receive_text(self):
        if not self.received:
            # Mimic client disconnect — raise to break the loop
            raise asyncio.CancelledError
        return self.received.pop(0)


class FakeFLServer:
    def __init__(self):
        class _Group:
            clients: dict = {}
            join_token = "secret-token"
            model_id = "x"

        class _GM:
            groups: dict = {}

            def register_client(self, client_id, group_id, client_info=None):
                return True

            def log_event(self, *args, **kwargs):
                pass

        class _CM:
            async def connect(self, ws):
                return None

            def register_client(self, cid, ws):
                return None

            def disconnect(self, ws):
                return None

        self.connection_manager = _CM()
        self.group_manager = _GM()
        self.group_manager.groups["g1"] = _Group()


@pytest.fixture
def stub_fl_server(monkeypatch):
    from astra.app import state

    state.set_fl_server(FakeFLServer())


@pytest.fixture
def valid_token(monkeypatch):
    """Stub the auth manager to accept any token and return a payload."""

    class _Payload:
        sub = "alice"
        user_id = 42

    from astra.infra.security.auth import get_auth_manager

    am = get_auth_manager()
    monkeypatch.setattr(am, "verify_token", lambda token: _Payload() if token else None)


def _run(ws):
    return websocket_endpoint(ws)


def test_rejects_train_command(stub_fl_server, valid_token):
    ws = FakeWebSocket(token="valid")
    ws.received.append(json.dumps({"type": "train_command", "config": {}}))
    with contextlib.suppress(asyncio.CancelledError):
        asyncio.run(_run(ws))
    assert any(
        m.get("status") == "rejected"
        and "client_training_no_longer_supported" in m.get("reason", "")
        for m in ws.sent
    ), ws.sent


def test_rejects_training_started(stub_fl_server, valid_token):
    ws = FakeWebSocket(token="valid")
    ws.received.append(json.dumps({"type": "training_started"}))
    with contextlib.suppress(asyncio.CancelledError):
        asyncio.run(_run(ws))
    assert any(
        m.get("status") == "rejected"
        and "client_training_no_longer_supported" in m.get("reason", "")
        for m in ws.sent
    )


def test_rejects_training_paused(stub_fl_server, valid_token):
    ws = FakeWebSocket(token="valid")
    ws.received.append(json.dumps({"type": "training_paused"}))
    with contextlib.suppress(asyncio.CancelledError):
        asyncio.run(_run(ws))
    assert any(
        m.get("status") == "rejected"
        and "client_training_no_longer_supported" in m.get("reason", "")
        for m in ws.sent
    )


def test_rejects_training_stopped(stub_fl_server, valid_token):
    ws = FakeWebSocket(token="valid")
    ws.received.append(json.dumps({"type": "training_stopped"}))
    with contextlib.suppress(asyncio.CancelledError):
        asyncio.run(_run(ws))
    assert any(
        m.get("status") == "rejected"
        and "client_training_no_longer_supported" in m.get("reason", "")
        for m in ws.sent
    )


def test_rejects_update(stub_fl_server, valid_token):
    ws = FakeWebSocket(token="valid")
    ws.received.append(json.dumps({"type": "update", "update": {"client_id": "c1"}}))
    with contextlib.suppress(asyncio.CancelledError):
        asyncio.run(_run(ws))
    assert any(
        m.get("status") == "rejected" and m.get("reason") == "updates_via_rest"
        for m in ws.sent
    )


def test_rejects_metrics(stub_fl_server, valid_token):
    ws = FakeWebSocket(token="valid")
    ws.received.append(json.dumps({"type": "metrics", "client_id": "c1"}))
    with contextlib.suppress(asyncio.CancelledError):
        asyncio.run(_run(ws))
    assert any(
        m.get("status") == "rejected"
        and "metrics_no_longer_supported" in m.get("reason", "")
        for m in ws.sent
    )


def test_rejects_missing_token(stub_fl_server, valid_token):
    ws = FakeWebSocket(token=None)
    asyncio.run(_run(ws))
    assert ws.closed_code == 1008


def test_rejects_invalid_token(stub_fl_server):
    ws = FakeWebSocket(token="bad")
    asyncio.run(_run(ws))
    assert ws.closed_code == 1008


def test_register_still_works(stub_fl_server, valid_token):
    ws = FakeWebSocket(token="valid")
    ws.received.append(
        json.dumps(
            {
                "type": "register",
                "client_id": "ws_c1",
                "group_id": "g1",
                "join_token": "secret-token",
            }
        )
    )
    with contextlib.suppress(asyncio.CancelledError):
        asyncio.run(_run(ws))
    # Registration should succeed (not be rejected)
    registered = [m for m in ws.sent if m.get("status") == "registered"]
    assert registered, ws.sent
    assert registered[0]["client_id"] == "ws_c1"
