"""
Shared server state for the networking package.

Provides a module-level accessor for the FLServer instance so that
route modules can access it without circular imports.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from networking.fl_server import FLServer

fl_server: "FLServer | None" = None


def get_fl_server() -> "FLServer":
    """Get the initialized FLServer instance. Raises if not yet initialized."""
    if fl_server is None:
        raise RuntimeError("FL Server not initialized")
    return fl_server
