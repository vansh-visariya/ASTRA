"""
Shared test fixtures.
"""

import contextlib
import os
import tempfile
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset


class TinyModel(torch.nn.Module):
    def __init__(self, input_dim=10, hidden_dim=5, num_classes=3):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


@pytest.fixture
def sample_config():
    return {
        "seed": 42,
        "dataset": {"name": "MNIST", "split": "iid"},
        "model": {},
        "client": {"num_clients": 5, "local_epochs": 1, "batch_size": 16, "lr": 0.01},
        "server": {
            "optimizer": "sgd",
            "server_lr": 0.5,
            "momentum": 0.9,
            "async_lambda": 0.2,
            "aggregator_window": 10,
            "adaptive_lr": False,
            "lr_decay_factor": 0.5,
            "instability_threshold": 0.15,
        },
        "robust": {"method": "fedavg", "trim_ratio": 0.1},
        "trust": {
            "init": 1.0,
            "update_alpha": 0.3,
            "quarantine_threshold": 0.35,
            "soft_decay": 0.8,
        },
        "malicious": {"enabled": False, "ratio": 0.0, "behaviors": []},
        "privacy": {
            "dp_enabled": False,
            "dp_mode": "client",
            "clip_norm": 1.0,
            "sigma": 1.2,
        },
        "communication": {"compression": "none"},
        "peft": {"enabled": False},
    }


def _make_tiny_config(**overrides):
    data = {
        "seed": 42,
        "dataset": {"name": "MNIST", "split": "iid"},
        "model": {},
        "client": {"num_clients": 5, "local_epochs": 1, "batch_size": 16, "lr": 0.01},
        "server": {
            "optimizer": "sgd",
            "server_lr": 0.5,
            "momentum": 0.9,
            "async_lambda": 0.2,
            "aggregator_window": 10,
            "adaptive_lr": False,
            "lr_decay_factor": 0.5,
            "instability_threshold": 0.15,
        },
        "robust": {"method": "fedavg", "trim_ratio": 0.1},
        "trust": {
            "init": 1.0,
            "update_alpha": 0.3,
            "quarantine_threshold": 0.35,
            "soft_decay": 0.8,
        },
        "malicious": {"enabled": False, "ratio": 0.0, "behaviors": []},
        "privacy": {
            "dp_enabled": False,
            "dp_mode": "client",
            "clip_norm": 1.0,
            "sigma": 1.2,
        },
        "communication": {"compression": "none"},
        "peft": {"enabled": False},
    }
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(data.get(k), dict):
            data[k].update(v)
        else:
            data[k] = v
    return data


@pytest.fixture
def dummy_data():
    x = torch.randn(100, 10)
    y = torch.randint(0, 3, (100,))
    return TensorDataset(x, y)


@pytest.fixture
def val_loader():
    x = torch.randn(20, 10)
    y = torch.randint(0, 3, (20,))
    return DataLoader(TensorDataset(x, y), batch_size=8)


@pytest.fixture
def tiny_model_factory():
    def _factory():
        return TinyModel()

    return _factory


# ----------------------------------------------------------------------
# Test-isolation fixtures: keep test-created rows out of the dev database.
#
# Problem: the test suite registers many disposable model_ids (info_*,
# dup_arch_*, jr2_*, arch_*, del_m_*, inv_m_*) and these rows end up
# visible in the dashboard's Model Configuration Registry view if the
# test process shares the developer's `astra.db`.
#
# Solution:
# 1. Session-scoped: redirect AstraDB to a fresh per-session file
#    (under the OS temp dir). Tests never touch the developer's real
#    `astra.db`. The file starts empty and is removed at session end.
# 2. Per-test: reset the in-memory model registry singleton and the
#    FLServer-attached registry. This keeps the FLServer's internal
#    state consistent across tests.
# 3. NO per-test DB truncation: the FLServer reloads the model
#    registry from the DB on its first delta upload (lazy init), and
#    truncating mid-session breaks that round-trip.
# ----------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def _redirect_db_to_temp_file():
    """Point the global AstraDB at a fresh per-session temp file."""
    from astra.app.database import AstraDB

    tmp_dir = Path(tempfile.gettempdir()) / "astra_tests"
    tmp_dir.mkdir(exist_ok=True)
    test_db_path = tmp_dir / "test_astra.db"
    if test_db_path.exists():
        test_db_path.unlink()
    os.environ["ASTRA_DB_PATH"] = str(test_db_path)
    AstraDB(str(test_db_path))
    yield
    # Best-effort cleanup of the temp file
    with contextlib.suppress(OSError):
        test_db_path.unlink()


def _reset_model_registry():
    """Reset the FLServer-attached model registry between tests.

    Why not reset `_global_registry` too? Tests that call
    `get_registry().register_factory(...)` register directly into the
    global registry, bypassing the API (and the DB persistence layer).
    If we wiped `_global_registry`, the FLServer's `_reload_models_from_db`
    on the next test's lifespan start wouldn't find those models —
    they'd be lost.

    Instead, we only clear the FLServer's attached registry so the next
    test's lifespan rebuilds it (from the global, which still has the
    previously-registered factories, AND from the test_astra.db).
    """
    with contextlib.suppress(Exception):
        from astra.app import state as _state

        fs = _state.fl_server
        if fs is not None:
            fs.model_registry = None


@pytest.fixture(autouse=True)
def _reset_registry():
    """Reset the FLServer's model registry between tests."""
    _reset_model_registry()
    yield
    _reset_model_registry()
