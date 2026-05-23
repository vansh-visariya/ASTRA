"""
Test fixtures for ASTRA.

Provides FastAPI TestClient with proper app setup for all test modules.
"""

import pytest


@pytest.fixture(autouse=True)
def _init_app_state():
    """Initialize FL server state before any test that hits the API."""
    import astra.app.state as state
    from astra.core.config import load_config
    from astra.app.fl_server import FLServer

    if state.fl_server is None:
        config = load_config()
        state.set_fl_server(FLServer(config))


@pytest.fixture
def client():
    from astra.app.server_api import app
    from fastapi.testclient import TestClient
    with TestClient(app) as c:
        yield c


@pytest.fixture
def sample_config():
    return {
        'seed': 42,
        'dataset': {'name': 'MNIST', 'split': 'iid'},
        'model': {'type': 'cnn', 'cnn': {'name': 'simple_cnn'}},
        'client': {'num_clients': 5, 'local_epochs': 1, 'batch_size': 16, 'lr': 0.01},
        'server': {
            'optimizer': 'sgd', 'server_lr': 0.5, 'momentum': 0.9,
            'async_lambda': 0.2, 'aggregator_window': 10,
            'adaptive_lr': False, 'lr_decay_factor': 0.5, 'instability_threshold': 0.15,
        },
        'robust': {'method': 'fedavg', 'trim_ratio': 0.1},
        'trust': {'init': 1.0, 'update_alpha': 0.3, 'quarantine_threshold': 0.35, 'soft_decay': 0.8},
        'malicious': {'enabled': False, 'ratio': 0.0, 'behaviors': []},
        'privacy': {'dp_enabled': False, 'dp_mode': 'client', 'clip_norm': 1.0, 'sigma': 1.2},
        'communication': {'compression': 'none', 'topk_ratio': 0.1},
        'training': {'total_steps': 100, 'eval_interval_steps': 10},
        'heterogeneous': {'mapping_method': 'average', 'allow_partial_updates': True, 'min_param_overlap': 0.5},
    }
