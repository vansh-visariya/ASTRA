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

# Load .env file before any other imports that read os.environ
_project_root = Path(_src_root).parent  # src/ -> repo root
_env_path = _project_root / ".env"
if _env_path.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env_path)
    except ImportError:
        pass  # python-dotenv not installed, user must set env vars manually

from contextlib import asynccontextmanager

import os

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import astra.app.state as state
from astra.app.fl_server import FLServer

# Route modules
from astra.app.routes import clients, experiments, groups, models, system
from astra.core.config import load_config
from astra.infra.websocket_handler import websocket_endpoint

# ============================================================================
# Lifespan
# ============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = load_config()

    fl_server = FLServer(config)
    fl_server.group_manager.server_model = fl_server.server.model if fl_server.server else None
    state.set_fl_server(fl_server)

    # Initialize the upload manager (presigned-URL flow for multi-GB deltas).
    secret_key = (
        config.get("secret_key")
        or os.environ.get("SECRET_KEY")
        or "astra-dev-secret"
    ).encode()
    from astra.app.uploads import init_upload_manager
    from astra.app.downloads import init_download_manager

    init_upload_manager(config=config, secret_key=secret_key)
    init_download_manager(config=config, secret_key=secret_key)

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
from astra.app.routes import uploads as uploads_routes
from astra.app.routes import downloads as downloads_routes
from astra.app.routes import announcements as announcements_routes
from astra.app.routes import messages as messages_routes

app.include_router(system.router)
app.include_router(groups.router)
app.include_router(clients.router)
app.include_router(models.router)
app.include_router(experiments.router)
app.include_router(uploads_routes.router)
app.include_router(downloads_routes.router)
app.include_router(announcements_routes.router)
app.include_router(messages_routes.router)

# WebSocket endpoint
app.websocket("/ws")(websocket_endpoint)


# ============================================================================
# Main
# ============================================================================


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the API server."""
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    run_server()
