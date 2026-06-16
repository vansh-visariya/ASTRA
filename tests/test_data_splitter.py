"""
Unit tests for DataSplitter: IID split, Dirichlet split, class distributions,
data loaders, and configuration handling.
"""

import numpy as np
import pytest

from astra.core.data_splitter import DataSplitter


@pytest.fixture
def base_config():
    return {
        "seed": 42,
        "dataset": {
            "name": "MNIST",
            "split": "dirichlet",
            "dirichlet_alpha": 0.3,
            "imbalance": True,
        },
        "client": {
            "num_clients": 10,
            "local_epochs": 2,
            "batch_size": 32,
        },
    }


class TestDataSplitterInit:
    def test_init_with_string_dataset(self):
        config = {"dataset": "MNIST", "client": {"num_clients": 5}}
        splitter = DataSplitter(config)
        assert splitter.dataset_name == "MNIST"
        assert splitter.num_clients == 5

    def test_init_with_dict_dataset(self, base_config):
        splitter = DataSplitter(base_config)
        assert splitter.dataset_name == "MNIST"
        assert splitter.num_clients == 10
        assert splitter.split_method == "dirichlet"

    def test_default_values(self):
        config = {}
        splitter = DataSplitter(config)
        assert splitter.dataset_name == "MNIST"
        assert splitter.split_method == "dirichlet"
        assert splitter.num_clients == 20


class TestIIDSplit:
    def test_iid_split_creates_all_clients(self, base_config):
        base_config["dataset"]["split"] = "iid"
        splitter = DataSplitter(base_config)
        splitter.split_data()
        assert len(splitter.client_data) == 10

    def test_iid_split_non_empty(self, base_config):
        base_config["dataset"]["split"] = "iid"
        splitter = DataSplitter(base_config)
        splitter.split_data()
        for client_id in range(10):
            assert len(splitter.client_data[client_id]) > 0

    def test_iid_split_total_samples(self, base_config):
        base_config["dataset"]["split"] = "iid"
        splitter = DataSplitter(base_config)
        splitter.split_data()
        total = sum(len(splitter.client_data[i]) for i in range(10))
        assert total == len(splitter.train_dataset)


class TestDirichletSplit:
    def test_dirichlet_split_creates_all_clients(self, base_config):
        splitter = DataSplitter(base_config)
        splitter.split_data()
        assert len(splitter.client_data) == 10

    def test_dirichlet_each_client_has_data(self, base_config):
        splitter = DataSplitter(base_config)
        splitter.split_data()
        for client_id in range(10):
            assert len(splitter.client_data[client_id]) > 0

    def test_dirichlet_total_equals_full(self, base_config):
        splitter = DataSplitter(base_config)
        splitter.split_data()
        total = sum(len(splitter.client_data[i]) for i in range(10))
        assert total == len(splitter.train_dataset)

    def test_dirichlet_class_distributions(self, base_config):
        splitter = DataSplitter(base_config)
        splitter.split_data()
        assert len(splitter.class_distributions) == 10
        probs = splitter.class_distributions[0].values()
        assert sum(probs) == pytest.approx(1.0, abs=0.01)


class TestCreateDataLoaders:
    def test_create_loaders_returns_correct_count(self, base_config):
        splitter = DataSplitter(base_config)
        client_loaders, val_loader = splitter.create_data_loaders()
        assert len(client_loaders) == 10

    def test_get_client_data(self, base_config):
        splitter = DataSplitter(base_config)
        data = splitter.get_client_data(0)
        assert len(data) > 0

    def test_get_class_distribution_default(self, base_config):
        splitter = DataSplitter(base_config)
        dist = splitter.get_class_distribution(999)
        assert dist == {}


class TestGetClientData:
    def test_get_client_data_triggers_split(self, base_config):
        splitter = DataSplitter(base_config)
        assert len(splitter.client_data) == 0
        _data = splitter.get_client_data(0)
        assert len(splitter.client_data) == 10


class TestInvalidConfig:
    def test_unknown_dataset_raises(self):
        config = {
            "dataset": {"name": "NONEXISTENT_DATASET"},
            "client": {"num_clients": 5},
        }
        with pytest.raises(ValueError):
            DataSplitter(config)

    def test_unknown_split_raises(self, base_config):
        base_config["dataset"]["split"] = "nonexistent_split"
        splitter = DataSplitter(base_config)
        with pytest.raises(ValueError):
            splitter.split_data()
