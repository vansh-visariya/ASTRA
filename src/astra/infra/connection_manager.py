"""
WebSocket connection manager for the Federated Learning server.

Handles WebSocket connections, broadcasting, and per-client messaging.
"""

import contextlib
import logging
from typing import Any

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections for live updates."""

    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self.client_sockets: dict[str, WebSocket] = {}
        self._websocket_to_client: dict[int, str] = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.debug("WebSocket connected (total: %s)", len(self.active_connections))

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        ws_id = id(websocket)
        client_id = self._websocket_to_client.pop(ws_id, None)
        if client_id and client_id in self.client_sockets:
            del self.client_sockets[client_id]
        else:
            client_id = client_id or "dashboard"
        logger.debug(
            "WebSocket disconnected: client=%s (active=%s, registered=%s)",
            client_id,
            len(self.active_connections),
            len(self.client_sockets),
        )

    async def broadcast(self, message: dict[str, Any]):
        """Broadcast message to all connected clients."""
        msg_type = message.get("type", "unknown")
        for connection in self.active_connections:
            with contextlib.suppress(Exception):
                await connection.send_json(message)
        logger.debug("Broadcast: type=%s to %s connections", msg_type, len(self.active_connections))

    async def send_to(self, client_id: str, message: dict[str, Any]):
        """Send message to specific client."""
        if client_id in self.client_sockets:
            with contextlib.suppress(Exception):
                await self.client_sockets[client_id].send_json(message)
        else:
            logger.debug("send_to: client %s not connected", client_id)

    def register_client(self, client_id: str, websocket: WebSocket):
        self.client_sockets[client_id] = websocket
        self._websocket_to_client[id(websocket)] = client_id
        logger.info(
            "Client registered: %s (total clients: %s)",
            client_id,
            len(self.client_sockets),
        )

    def unregister_client(self, client_id: str):
        ws = self.client_sockets.pop(client_id, None)
        if ws is not None:
            self._websocket_to_client.pop(id(ws), None)
        logger.info("Client unregistered: %s (remaining: %s)", client_id, len(self.client_sockets))
