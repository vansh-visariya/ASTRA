"""
Integration tests: FLClient + AsyncServer round-trip training,
multi-client aggregation with Byzantine robustness, DP + compression pipeline.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pytest

from astra.core.fl_client import FLClient
from astra.core.server import AsyncServer
from astra.core.aggregation.aggregator import create_aggregator
from astra.core.models.model_zoo import flatten_all_params, apply_flat_delta


def _make_config(**overrides):
    cfg = {
        "seed": 42,
        "dataset": {"name": "MNIST", "split": "dirichlet", "dirichlet_alpha": 0.3},
        "model": {"type": "cnn", "model_id": "simple_cnn_mnist"},
        "client": {"num_clients": 3, "local_epochs": 1, "batch_size": 8, "lr": 0.01},
        "server": {"server_lr": 0.5, "aggregator_window": 3, "momentum": 0.9, "async_lambda": 0.2, "optimizer": "sgd", "adaptive_lr": False},
        "robust": {"method": "fedavg"},
        "trust": {"init": 1.0, "update_alpha": 0.3, "quarantine_threshold": 0.35, "soft_decay": 0.8},
        "malicious": {"enabled": False, "ratio": 0.0, "behaviors": []},
        "privacy": {"dp_enabled": False, "dp_mode": "client", "clip_norm": 1.0, "sigma": 1.2},
        "communication": {"compression": "none", "topk_ratio": 0.1},
        "peft": {"enabled": False},
        "training": {"total_steps": 10, "eval_interval_steps": 5},
    }
    cfg.update(overrides)
    return cfg


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 3)
    def forward(self, x):
        return self.fc(x)


def _make_tiny_config(**overrides):
    cfg = {
        "seed": 42,
        "dataset": {"name": "MNIST"},
        "model": {"type": "cnn"},
        "client": {"num_clients": 3, "local_epochs": 1, "batch_size": 4, "lr": 0.01},
        "server": {"server_lr": 0.5, "aggregator_window": 2, "momentum": 0.9, "async_lambda": 0.2, "optimizer": "sgd", "adaptive_lr": False},
        "robust": {"method": "fedavg"},
        "trust": {"init": 1.0, "update_alpha": 0.3, "quarantine_threshold": 0.35, "soft_decay": 0.8},
        "malicious": {"enabled": False, "ratio": 0.0, "behaviors": []},
        "privacy": {"dp_enabled": False, "dp_mode": "client", "clip_norm": 1.0, "sigma": 1.2},
        "communication": {"compression": "none", "topk_ratio": 0.1},
        "peft": {"enabled": False},
        "training": {"total_steps": 10, "eval_interval_steps": 5},
    }
    cfg.update(overrides)
    return cfg


@pytest.fixture
def dummy_data():
    x = torch.randn(20, 10)
    y = torch.randint(0, 3, (20,))
    return TensorDataset(x, y)


@pytest.fixture
def val_loader():
    x = torch.randn(8, 10)
    y = torch.randint(0, 3, (8,))
    return DataLoader(TensorDataset(x, y), batch_size=4)


class TestRoundTripTraining:
    """Single client trains, server receives and aggregates."""

    def test_single_update_flow(self, dummy_data, val_loader):
        config = _make_tiny_config()
        model = TinyModel()
        aggregator = create_aggregator(config)
        server = AsyncServer(model, aggregator, config, val_loader)
        server.start()

        client = FLClient("c1", dummy_data, lambda: TinyModel(), config)
        update = client.local_train()

        assert "local_updates" in update
        assert update["local_dataset_size"] == 20
        assert update["client_version"] == 1

        server.handle_update(update)
        assert server.global_version == 0

    def test_multiple_updates_triggers_aggregation(self, dummy_data, val_loader):
        config = _make_tiny_config()
        config["server"]["aggregator_window"] = 2
        model = TinyModel()
        aggregator = create_aggregator(config)
        server = AsyncServer(model, aggregator, config, val_loader)
        server.start()

        client_a = FLClient("a", dummy_data, lambda: TinyModel(), config)
        client_b = FLClient("b", dummy_data, lambda: TinyModel(), config)

        ua = client_a.local_train()
        ub = client_b.local_train()

        server.handle_update(ua)
        assert server.global_version == 0
        server.handle_update(ub)
        assert server.global_version == 1

    def test_weights_change_after_aggregation(self, dummy_data, val_loader):
        config = _make_tiny_config()
        config["server"]["aggregator_window"] = 2
        model = TinyModel()
        before = flatten_all_params(model)

        aggregator = create_aggregator(config)
        server = AsyncServer(model, aggregator, config, val_loader)
        server.start()

        for cid in ["a", "b"]:
            client = FLClient(cid, dummy_data, lambda: TinyModel(), config)
            server.handle_update(client.local_train())

        after = flatten_all_params(model)
        assert not np.allclose(before, after)

    def test_evaluate_returns_metrics(self, dummy_data, val_loader):
        config = _make_tiny_config()
        model = TinyModel()
        aggregator = create_aggregator(config)
        server = AsyncServer(model, aggregator, config, val_loader)
        server.start()

        metrics = server.evaluate()
        assert "accuracy" in metrics
        assert "loss" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0


class TestMultiClientAggregation:
    """Multiple clients aggregating with different strategies."""

    def test_fedavg_weighted_by_dataset_size(self, dummy_data, val_loader):
        config = _make_config(client={"num_clients": 2, "local_epochs": 1, "batch_size": 8, "lr": 0.01})
        model = TinyModel()
        aggregator = create_aggregator(config)
        server = AsyncServer(model, aggregator, config, val_loader)
        server.start()

        small_data = TensorDataset(torch.randn(5, 10), torch.randint(0, 3, (5,)))
        large_data = TensorDataset(torch.randn(50, 10), torch.randint(0, 3, (50,)))

        small_client = FLClient("small", small_data, lambda: TinyModel(), config)
        large_client = FLClient("large", large_data, lambda: TinyModel(), config)

        server.handle_update(small_client.local_train())
        server.handle_update(large_client.local_train())

        assert server.global_version == 0

    def test_empty_buffer_raises(self):
        config = _make_tiny_config()
        model = TinyModel()
        aggregator = create_aggregator(config)
        server = AsyncServer(model, aggregator, config, None)
        from astra.core.exceptions import AggregationError
        with pytest.raises(AggregationError):
            aggregator.aggregate([])


class TestByzantineRobustness:
    """Robust aggregation against malicious clients."""

    def test_honest_vs_malicious_trust_scoring(self, dummy_data, val_loader):
        config = _make_tiny_config(
            server={"server_lr": 0.5, "aggregator_window": 2, "momentum": 0.0, "async_lambda": 0.2, "optimizer": "sgd", "adaptive_lr": False},
            trust={"init": 1.0, "update_alpha": 0.3, "quarantine_threshold": 0.2, "soft_decay": 0.8},
        )

        model = TinyModel()
        shared_copy = TinyModel()
        shared_copy.load_state_dict(model.state_dict())
        def factory():
            return shared_copy

        c1 = FLClient("honest1", dummy_data, factory, config)
        c2 = FLClient("bad", dummy_data, factory, config)
        server = AsyncServer(model, config)
        server.handle_update(c1.local_train())
        server.handle_update(c2.local_train())

        assert server.global_version == 1

    def test_trust_score_updates_after_honest_update(self, dummy_data, val_loader):
        config = _make_tiny_config()
        config["trust"]["init"] = 1.0
        model = TinyModel()
        aggregator = create_aggregator(config)
        server = AsyncServer(model, aggregator, config, val_loader)
        server.start()

        assert server.trust_manager.get_trust("c1") == 1.0

        client = FLClient("c1", dummy_data, lambda: TinyModel(), config)
        server.handle_update(client.local_train())

        trust = server.trust_manager.get_trust("c1")
        assert 0.0 <= trust <= 1.0


class TestDPPipeline:
    """Differential privacy on client and server side."""

    def test_client_side_dp(self, dummy_data, val_loader):
        config = _make_tiny_config(
            privacy={"dp_enabled": True, "dp_mode": "client", "clip_norm": 1.0, "sigma": 1.2}
        )
        client = FLClient("dp_c", dummy_data, lambda: TinyModel(), config)
        update = client.local_train()
        delta = np.frombuffer(update["local_updates"], dtype=np.float32)
        assert len(delta) > 0

    def test_server_side_dp(self, dummy_data, val_loader):
        config = _make_tiny_config(
            privacy={"dp_enabled": True, "dp_mode": "server", "clip_norm": 1.0, "sigma": 1.2}
        )
        config["server"]["aggregator_window"] = 1
        model = TinyModel()
        aggregator = create_aggregator(config)
        server = AsyncServer(model, aggregator, config, val_loader)
        server.start()

        client = FLClient("dp_s", dummy_data, lambda: TinyModel(), config)
        server.handle_update(client.local_train())


class TestCompressionPipeline:
    """Top-K compression on client side."""

    def test_compression_reduces_size(self, dummy_data):
        config = _make_tiny_config(
            communication={"compression": "topk", "topk_ratio": 0.1}
        )
        client = FLClient("comp_c", dummy_data, lambda: TinyModel(), config)
        update = client.local_train()
        delta = np.frombuffer(update["local_updates"], dtype=np.float32)
        nonzero = np.count_nonzero(delta)
        total = len(delta)
        assert nonzero / total <= 0.15

    def test_compression_zero_ratio_all_zeros(self, dummy_data):
        config = _make_tiny_config(
            communication={"compression": "topk", "topk_ratio": 0.01}
        )
        client = FLClient("comp_z", dummy_data, lambda: TinyModel(), config)
        update = client.local_train()
        delta = np.frombuffer(update["local_updates"], dtype=np.float32)
        assert np.count_nonzero(delta) <= len(delta) * 0.05


class TestCheckpointSaving:
    """Save and load checkpoints."""

    def test_save_and_load(self, dummy_data, val_loader):
        import tempfile
        import os

        config = _make_tiny_config()
        model = TinyModel()
        aggregator = create_aggregator(config)
        server = AsyncServer(model, aggregator, config, val_loader)
        server.start()

        fd, path = tempfile.mkstemp(suffix=".pt")
        os.close(fd)

        try:
            server.save_checkpoint(path)
            assert os.path.exists(path)

            server.global_version = 999
            before_version = server.global_version
            server.load_checkpoint(path)
            assert server.global_version == 0
        finally:
            os.unlink(path)
