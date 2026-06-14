"""
FastAPI Server for Distributed Federated Learning.

This is the application entry point that assembles the FastAPI app
from the modular route, WebSocket, and server components.

Provides:
- REST API for training control
- WebSocket for live updates
- Client registration and management
- Group-based training with hybrid async windowing
- Experiment tracking with SQLite
"""

import sys
from pathlib import Path

# Add src/ to path so `import astra` resolves to src/astra/
_src_root = Path(__file__).parent.parent.parent  # src/astra/app/ -> src/
if str(_src_root) not in sys.path:
    sys.path.insert(0, str(_src_root))

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi_socketio import SocketManager

import astra.app.state as state
from astra.app.fl_server import FLServer

# Route modules
from astra.app.routes import clients, experiments, groups, models, system
from astra.core.config import load_config
from astra.infra.websocket_handler import register_socketio_handlers, websocket_endpoint

# ============================================================================
# Lifespan
# ============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = load_config()

    fl_server = FLServer(config)
    fl_server.group_manager.server_model = fl_server.server.model if fl_server.server else None
    state.set_fl_server(fl_server)

    yield

    if state.fl_server:
        state.fl_server.stop_experiment()


# ============================================================================
# Extended API registration
# ============================================================================

_extended_api_registered = False


def _register_extended_endpoints(app, config):
    """Register extended API endpoints."""
    global _extended_api_registered
    if _extended_api_registered:
        return

    try:
        from astra.app.extended_endpoints import setup_extended_api

        setup_extended_api(app, config)
        print("[INFO] Extended API endpoints registered")
        _extended_api_registered = True
    except Exception as e:
        print(f"[WARN] Could not register extended endpoints: {e}")


# ============================================================================
# App assembly
# ============================================================================

app = FastAPI(title="Federated Learning API", lifespan=lifespan)

# Register extended API endpoints at module level (auth, join requests, notifications, etc.)
# These endpoints don't depend on fl_server — they use their own FLPlatformIntegration.
_register_extended_endpoints(app, {})

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include route modules
app.include_router(system.router)
app.include_router(groups.router)
app.include_router(clients.router)
app.include_router(models.router)
app.include_router(experiments.router)

# WebSocket endpoint
app.websocket("/ws")(websocket_endpoint)

# Socket.IO support
socket_manager = SocketManager(app, cors_allowed_origins="*")
register_socketio_handlers(socket_manager)


# ============================================================================
# Main
# ============================================================================


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the API server."""
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    run_server()
