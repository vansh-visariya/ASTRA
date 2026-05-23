"""Core FL algorithms — pure Python, no web dependencies."""

__all__ = [
    "AsyncServer",
    "FLClient",
    "TrustManager",
    "DataSplitter",
    "load_config",
]


def __getattr__(name):
    if name == "AsyncServer":
        from astra.core.server import AsyncServer
        return AsyncServer
    if name == "FLClient":
        from astra.core.fl_client import FLClient
        return FLClient
    if name == "TrustManager":
        from astra.core.trust_manager import TrustManager
        return TrustManager
    if name == "DataSplitter":
        from astra.core.data_splitter import DataSplitter
        return DataSplitter
    if name == "load_config":
        from astra.core.config import load_config
        return load_config
    raise AttributeError(f"module 'astra.core' has no attribute {name!r}")
