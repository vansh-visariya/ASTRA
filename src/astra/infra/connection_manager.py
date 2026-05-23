"""
WebSocket connection manager for the Federated Learning server.

Handles WebSocket connections, broadcasting, and per-client messaging.
"""

from typing import Any, Dict, List
from fastapi import WebSocket


class ConnectionManager:
    """Manages WebSocket connections for live updates."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.client_sockets: Dict[str, WebSocket] = {}
        self._websocket_to_client: Dict[int, str] = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        ws_id = id(websocket)
        client_id = self._websocket_to_client.pop(ws_id, None)
        if client_id and client_id in self.client_sockets:
            del self.client_sockets[client_id]

    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients."""
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                pass

    async def send_to(self, client_id: str, message: Dict[str, Any]):
        """Send message to specific client."""
        if client_id in self.client_sockets:
            try:
                await self.client_sockets[client_id].send_json(message)
            except Exception:
                pass

    def register_client(self, client_id: str, websocket: WebSocket):
        self.client_sockets[client_id] = websocket
        self._websocket_to_client[id(websocket)] = client_id

    def unregister_client(self, client_id: str):
        ws = self.client_sockets.pop(client_id, None)
        if ws is not None:
            self._websocket_to_client.pop(id(ws), None)
