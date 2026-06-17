"""
ASTRA — Async Scalable Training & Research Architecture.

A production-ready distributed Federated Learning platform.
Clients train externally and upload model deltas to the server,
which aggregates them and broadcasts the new global model.
"""

__all__ = [
    "AsyncServer",
    "TrustManager",
    "load_config",
]


def __getattr__(name):
    if name == "AsyncServer":
        from astra.core.server import AsyncServer

        return AsyncServer
    if name == "TrustManager":
        from astra.core.trust_manager import TrustManager

        return TrustManager
    if name == "load_config":
        from astra.core.config import load_config

        return load_config
    raise AttributeError(f"module 'astra' has no attribute {name!r}")
