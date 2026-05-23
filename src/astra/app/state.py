"""
Shared server state.

Provides a module-level accessor for the FLServer instance so that
route modules can access it without circular imports.

The lifespan in server_api.py sets state.fl_server during startup.
For testing, assign state.fl_server directly to a mock/test instance.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from astra.app.fl_server import FLServer

fl_server: "FLServer | None" = None


def set_fl_server(server: "FLServer") -> None:
    """Called by the application lifespan to register the server instance."""
    global fl_server
    fl_server = server


def get_fl_server() -> "FLServer":
    """Get the initialized FLServer instance. Raises if not yet initialized."""
    if fl_server is None:
        raise RuntimeError(
            "FL Server not initialized. The application lifespan must "
            "call set_fl_server() during startup."
        )
    return fl_server
