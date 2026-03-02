"""
Configuration models using Pydantic for validation.

Provides type-safe configuration with validation.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class ClientConfig(BaseModel):
    """Client-side configuration."""

    num_clients: int = Field(default=20, ge=1)
    local_epochs: int = Field(default=2, ge=1)
    batch_size: int = Field(default=32, ge=1)
    lr: float = Field(default=0.01, gt=0)
    weight_decay: float = Field(default=0.0, ge=0)


class ServerConfig(BaseModel):
    """Server-side configuration."""

    optimizer: Literal["sgd", "adam"] = "sgd"
    server_lr: float = Field(default=0.5, gt=0)
    momentum: float = Field(default=0.9, ge=0)
    async_lambda: float = Field(default=0.2, ge=0)
    aggregator_window: int = Field(default=10, ge=1)
    poll_timeout: float = Field(default=1.0, gt=0)
    min_agg_interval_seconds: float = Field(default=0.5, gt=0)
    adaptive_lr: bool = False
    lr_decay_factor: float = Field(default=0.5, gt=0, le=1)
    instability_threshold: float = Field(default=0.15, gt=0)


class RobustConfig(BaseModel):
    """Robust aggregation configuration."""

    method: Literal["fedavg", "median", "trimmed_mean", "hybrid"] = "hybrid"
    trim_ratio: float = Field(default=0.1, ge=0, le=0.5)
    norm_clip: float = Field(default=5.0, gt=0)
    anomaly_k: float = Field(default=3.0, gt=0)
    sim_threshold: float = Field(default=0.2, ge=0, le=1)
    trust_power: float = Field(default=1.0, ge=0)


class TrustConfig(BaseModel):
    """Trust management configuration."""

    init: float = Field(default=1.0, ge=0, le=1)
    update_alpha: float = Field(default=0.3, gt=0, le=1)
    quarantine_threshold: float = Field(default=0.35, ge=0, le=1)
    soft_decay: float = Field(default=0.8, gt=0, le=1)


class MaliciousConfig(BaseModel):
    """Malicious client simulation configuration."""

    ratio: float = Field(default=0.0, ge=0, le=1)
    behaviors: list[str] = Field(default_factory=lambda: ["label_flip", "noise"])


class PrivacyConfig(BaseModel):
    """Differential privacy configuration."""

    dp_enabled: bool = False
    dp_mode: Literal["client", "server"] = "client"
    clip_norm: float = Field(default=1.0, gt=0)
    sigma: float = Field(default=1.2, gt=0)
    epsilon_target: float | None = None


class CommunicationConfig(BaseModel):
    """Communication compression configuration."""

    compression: Literal["none", "topk", "quantize"] = "topk"
    topk_ratio: float = Field(default=0.1, gt=0, le=1)
    quantize_bits: int = Field(default=8, ge=1, le=32)


class TrainingConfig(BaseModel):
    """Training configuration."""

    total_steps: int = Field(default=2000, ge=1)
    eval_interval_steps: int = Field(default=50, ge=1)


class LoggingConfig(BaseModel):
    """Logging configuration."""

    log_dir: str = "./runs/exp1"
    jsonl_logging: bool = True


class DatasetConfig(BaseModel):
    """Dataset configuration."""

    name: Literal["MNIST", "CIFAR10"] = "MNIST"
    split: Literal["iid", "dirichlet"] = "dirichlet"
    dirichlet_alpha: float = Field(default=0.3, gt=0)
    imbalance: bool = True


class ModelConfig(BaseModel):
    """Model configuration."""

    type: Literal["cnn", "hf_transformer"] = "cnn"
    cnn: dict[str, Any] = Field(default_factory=lambda: {"name": "simple_cnn"})
    hf: dict[str, Any] = Field(default_factory=dict)


class PEFTConfig(BaseModel):
    """PEFT configuration."""

    enabled: bool = False
    method: Literal["lora", "adalora"] = "lora"
    lora_rank: int = Field(default=8, ge=1)
    lora_alpha: float = Field(default=16, gt=0)
    target_modules: list[str] = Field(default_factory=lambda: ["q_proj", "v_proj"])


class FLConfig(BaseModel):
    """Main Federated Learning configuration."""

    seed: int = 20260221
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    peft: PEFTConfig = Field(default_factory=PEFTConfig)
    client: ClientConfig = Field(default_factory=ClientConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    robust: RobustConfig = Field(default_factory=RobustConfig)
    trust: TrustConfig = Field(default_factory=TrustConfig)
    malicious: MaliciousConfig = Field(default_factory=MaliciousConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)
    communication: CommunicationConfig = Field(default_factory=CommunicationConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @field_validator("dataset", mode="before")
    @classmethod
    def parse_dataset(cls, v: Any) -> DatasetConfig:
        if isinstance(v, dict):
            return DatasetConfig(**v)
        return v

    @field_validator("model", mode="before")
    @classmethod
    def parse_model(cls, v: Any) -> ModelConfig:
        if isinstance(v, dict):
            return ModelConfig(**v)
        return v

    @field_validator("client", mode="before")
    @classmethod
    def parse_client(cls, v: Any) -> ClientConfig:
        if isinstance(v, dict):
            return ClientConfig(**v)
        return v

    @field_validator("server", mode="before")
    @classmethod
    def parse_server(cls, v: Any) -> ServerConfig:
        if isinstance(v, dict):
            return ServerConfig(**v)
        return v


def validate_config(config: dict[str, Any]) -> FLConfig:
    """
    Validate and parse configuration dictionary.

    Args:
        config: Raw configuration dictionary.

    Returns:
        Validated FLConfig instance.
    """
    return FLConfig(**config)
