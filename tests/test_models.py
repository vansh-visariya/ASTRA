"""
Unit tests for Pydantic request/response models: validation, defaults, serialization.
"""

import base64
import json

import pytest
from pydantic import ValidationError

from astra.infra.models import (
    ClientRegister,
    ClientUpdate,
    ExperimentConfig,
    ControlCommand,
)


class TestClientRegister:
    def test_minimal(self):
        m = ClientRegister(client_id="c1")
        assert m.client_id == "c1"
        assert m.capabilities == {}

    def test_with_capabilities(self):
        m = ClientRegister(client_id="c1", capabilities={"gpu": True, "cores": 4})
        assert m.capabilities["gpu"] is True
        assert m.capabilities["cores"] == 4

    def test_missing_client_id(self):
        with pytest.raises(ValidationError):
            ClientRegister()


class TestClientUpdate:
    def test_valid(self):
        delta = base64.b64encode(b"simulated_delta_data").decode()
        m = ClientUpdate(
            client_id="c1",
            client_version=1,
            local_updates=delta,
            local_dataset_size=1000,
        )
        assert m.client_id == "c1"
        assert m.client_version == 1
        assert m.update_type == "delta"

    def test_defaults(self):
        delta = base64.b64encode(b"data").decode()
        m = ClientUpdate(
            client_id="c2",
            client_version=0,
            local_updates=delta,
            local_dataset_size=500,
        )
        assert m.meta == {}
        assert m.update_type == "delta"

    def test_with_metadata(self):
        delta = base64.b64encode(b"data").decode()
        m = ClientUpdate(
            client_id="c3",
            client_version=2,
            local_updates=delta,
            local_dataset_size=2000,
            meta={"loss": 0.5, "accuracy": 0.85},
        )
        assert m.meta["loss"] == 0.5
        assert m.meta["accuracy"] == 0.85

    def test_missing_required(self):
        with pytest.raises(ValidationError):
            ClientUpdate()

    def test_invalid_version_type(self):
        with pytest.raises(ValidationError):
            ClientUpdate(
                client_id="c1",
                client_version="not_an_int",
                local_updates=base64.b64encode(b"x").decode(),
                local_dataset_size=100,
            )

    def test_serialization_roundtrip(self):
        delta = base64.b64encode(b"test_data_here").decode()
        m = ClientUpdate(
            client_id="c4",
            client_version=5,
            local_updates=delta,
            local_dataset_size=3000,
            meta={"key": "value"},
        )
        d = m.model_dump()
        m2 = ClientUpdate(**d)
        assert m2.client_id == m.client_id
        assert m2.client_version == m.client_version
        assert m2.local_updates == m.local_updates
        assert m2.local_dataset_size == m.local_dataset_size


class TestExperimentConfig:
    def test_valid(self):
        m = ExperimentConfig(experiment_id="exp1", config={"lr": 0.01})
        assert m.experiment_id == "exp1"
        assert m.config["lr"] == 0.01

    def test_empty_config(self):
        m = ExperimentConfig(experiment_id="exp2", config={})
        assert m.config == {}

    def test_missing_fields(self):
        with pytest.raises(ValidationError):
            ExperimentConfig()


class TestControlCommand:
    def test_start(self):
        m = ControlCommand(command="start")
        assert m.command == "start"
        assert m.params == {}

    def test_with_params(self):
        m = ControlCommand(command="resume", params={"from_step": 100})
        assert m.command == "resume"
        assert m.params["from_step"] == 100

    def test_invalid_command_type(self):
        with pytest.raises(ValidationError):
            ControlCommand(command="start", params="not_a_dict")

    def test_missing_command(self):
        with pytest.raises(ValidationError):
            ControlCommand()

    def test_all_commands(self):
        for cmd in ["start", "pause", "resume", "stop"]:
            m = ControlCommand(command=cmd)
            assert m.command == cmd

    def test_serialization(self):
        m = ControlCommand(command="stop", params={"reason": "done"})
        d = m.model_dump()
        m2 = ControlCommand(**d)
        assert m2.command == "stop"
        assert m2.params["reason"] == "done"
