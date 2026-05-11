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

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from aiohttp import web
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi_socketio import SocketManager

from api.database import init_db
from core_engine.server import AsyncServer

import networking.state as state
from networking.fl_server import FLServer
from networking.websocket_handler import websocket_endpoint, register_socketio_handlers

# Route modules
from networking.routes import system, groups, clients, models, experiments


# ============================================================================
# Lifespan
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    config = {
        'seed': 42,
        'dataset': {'name': 'MNIST', 'split': 'dirichlet', 'dirichlet_alpha': 0.3},
        'model': {'type': 'cnn', 'cnn': {'name': 'simple_cnn'}},
        'client': {'num_clients': 10, 'local_epochs': 2, 'batch_size': 32, 'lr': 0.01},
        'server': {'optimizer': 'sgd', 'server_lr': 0.5, 'momentum': 0.9, 'async_lambda': 0.2, 'aggregator_window': 5},
        'robust': {'method': 'fedavg', 'trim_ratio': 0.1},
        'privacy': {'dp_enabled': False},
        'training': {'total_steps': 1000, 'eval_interval_steps': 10},
        'heterogeneous': {'mapping_method': 'average', 'allow_partial_updates': True, 'min_param_overlap': 0.5},
    }

    state.fl_server = FLServer(config)

    # Setup Socket.IO
    socketio_app = web.Application()

    # Note: time-based aggregation is handled by per-group _training_watchdog

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
        from api.extended_endpoints import setup_extended_api
        platform = setup_extended_api(app, config)
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
