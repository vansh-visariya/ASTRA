"""Core FL algorithms — pure Python, no web dependencies.

The server-side FL engine: aggregation, trust scoring, model utilities,
and privacy/compression primitives. Clients train externally and submit
pre-computed deltas via the REST API.
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
    raise AttributeError(f"module 'astra.core' has no attribute {name!r}")
