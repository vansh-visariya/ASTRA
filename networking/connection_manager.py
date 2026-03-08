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

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

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

    def unregister_client(self, client_id: str):
        if client_id in self.client_sockets:
            del self.client_sockets[client_id]
