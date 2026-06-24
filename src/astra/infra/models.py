"""
Pydantic request/response models for the Federated Learning API.
"""

from typing import Any

from pydantic import BaseModel, Field


class ClientRegister(BaseModel):
    client_id: str
    capabilities: dict[str, Any] = {}


class ClientUpdate(BaseModel):
    client_id: str
    client_version: int
    local_updates: str  # Base64 encoded
    update_type: str = "delta"
    local_dataset_size: int
    meta: dict[str, Any] = {}


class ExperimentConfig(BaseModel):
    experiment_id: str
    config: dict[str, Any]


class ControlCommand(BaseModel):
    command: str  # start, pause, resume, stop
    params: dict[str, Any] = {}


class TrainingManifest(BaseModel):
    """Training contract between admin and clients.

    The admin defines this when creating a group. Clients must conform
    to these constraints when training and uploading deltas.
    """

    # --- Contract versioning ---
    contract_version: int = Field(
        1, description="Contract version. Bumped on each update so clients know if their training config is stale."
    )

    # --- Architecture (advisory — for client reference) ---
    model_id: str = Field(..., description="Model identifier matching the group's model_id")
    is_peft: bool = Field(False, description="Whether the group uses PEFT/LoRA adapters")
    target_modules: list[str] | None = Field(
        None,
        description="LoRA target modules (e.g. ['q_proj', 'v_proj']). "
        "Required when is_peft=True.",
    )
    lora_rank: int | None = Field(None, description="LoRA rank. Required when is_peft=True.")
    lora_alpha: float | None = Field(None, description="LoRA alpha. Required when is_peft=True.")
    expected_delta_bytes: int | None = Field(
        None, description="Advisory delta payload size in bytes (float32). "
        "Not enforced — client sizes may vary based on data and training config."
    )

    # --- Training protocol (advisory — for client reference) ---
    lr: float = Field(0.01, description="Recommended learning rate")
    batch_size: int = Field(32, description="Recommended batch size")
    local_epochs: int = Field(2, description="Recommended local training epochs")
    optimizer: str = Field("adamw", description="Recommended optimizer (e.g. 'adamw', 'sgd')")
    loss_function: str = Field(
        "cross_entropy", description="Recommended loss function"
    )
    max_grad_norm: float | None = Field(
        None, description="Recommended gradient clipping norm"
    )

    # --- Data schema (informational — for client reference) ---
    input_features: list[str] | None = Field(
        None, description="Expected input feature names"
    )
    input_shape: list[int] | None = Field(
        None, description="Expected input tensor shape (e.g. [128] for 1D, [3, 224, 224] for images)"
    )
    num_classes: int | None = Field(None, description="Number of output classes")
    label_type: str | None = Field(
        None, description="Label type: 'classification', 'regression', 'causal_lm'"
    )
    data_description: str | None = Field(
        None, description="Free-text description of the training data format and contents"
    )
    preprocessing_steps: list[str] | None = Field(
        None, description="Expected preprocessing steps (e.g. ['normalize', 'tokenize'])"
    )
    accepted_update_types: list[str] | None = Field(
        None, description="Accepted update types (e.g. ['delta', 'adapter'])"
    )

    # --- Verification metadata ---
    val_dataset: str | None = Field(
        None, description="Path to admin-uploaded validation dataset (.pt file with X/y tensors)"
    )
    val_metric: str = Field(
        "accuracy",
        description="Primary metric for server-side evaluation: 'accuracy', 'f1', 'mse'",
    )
