"""Application layer — API, orchestration, DB, group lifecycle."""

__all__ = [
    "FLServer",
    "GroupManager",
    "TrainingGroup",
    "AsyncWindowConfig",
    "get_fl_server",
    "set_fl_server",
]


def __getattr__(name):
    if name == "FLServer":
        from astra.app.fl_server import FLServer
        return FLServer
    if name == "GroupManager":
        from astra.app.group_manager import GroupManager
        return GroupManager
    if name in ("TrainingGroup", "AsyncWindowConfig"):
        from astra.app import training_group
        return getattr(training_group, name)
    if name == "get_fl_server":
        from astra.app.state import get_fl_server
        return get_fl_server
    if name == "set_fl_server":
        from astra.app.state import set_fl_server
        return set_fl_server
    raise AttributeError(f"module 'astra.app' has no attribute {name!r}")
