"""Infrastructure — transport, DB schemas, auth, WebSocket, model registry."""

__all__ = [
    "ConnectionManager",
    "ModelRegistry",
    "get_registry",
    "AuthManager",
    "get_auth_manager",
]


def __getattr__(name):
    if name == "ConnectionManager":
        from astra.infra.connection_manager import ConnectionManager

        return ConnectionManager
    if name in ("ModelRegistry", "get_registry"):
        from astra.infra import registry

        return getattr(registry, name)
    if name in ("AuthManager", "get_auth_manager"):
        from astra.infra.security import auth

        return getattr(auth, name)
    raise AttributeError(f"module 'astra.infra' has no attribute {name!r}")
