"""
Model management REST endpoints.
"""

import importlib
import json
import os
from typing import Any

import requests
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from astra.app.database import get_db
from astra.app.state import get_fl_server

router = APIRouter()


class RegisterArchitectureBody(BaseModel):
    model_id: str
    architecture_path: str
    model_type: str = "vision"
    config: dict | None = None


class RegisterHfBody(BaseModel):
    model_name: str
    use_peft: bool = False
    peft_method: str = "lora"


@router.get("/api/models")
async def list_models():
    """List all available models."""
    fl_server = get_fl_server()
    models = fl_server.model_registry.list_models()
    return {"models": models, "count": len(models)}


def _fetch_hf_model_metadata(model_name: str) -> dict[str, Any]:
    """Fetch lightweight metadata from HuggingFace for dataset sizing."""
    try:
        url = f"https://huggingface.co/api/models/{model_name}"
        res = requests.get(url, timeout=5)
        if res.status_code != 200:
            return {}
        return res.json() or {}
    except Exception:
        return {}


@router.post("/api/models/register/hf")
async def register_hf_model(body: RegisterHfBody):
    """Register a HuggingFace model."""
    fl_server = get_fl_server()
    try:
        peft_config = (
            {
                "enabled": body.use_peft,
                "method": body.peft_method,
                "lora_rank": 8,
                "lora_alpha": 16,
                "target_modules": ["q_proj", "v_proj"],
            }
            if body.use_peft
            else {"enabled": False}
        )

        model_info = fl_server.model_registry.register_hf_model(
            model_name=body.model_name, use_peft=body.use_peft, peft_config=peft_config
        )

        hf_meta = _fetch_hf_model_metadata(body.model_name)
        hf_config = hf_meta.get("config") or {}
        vision_config = hf_config.get("vision_config") or {}
        image_size = hf_config.get("image_size") or vision_config.get("image_size")
        if image_size:
            model_info.config.setdefault("dataset", {})
            model_info.config["dataset"].setdefault("image_size", image_size)
            model_info.config["dataset"].setdefault("channels", 3)
            model_info.config["dataset"].setdefault(
                "normalize_mean", (0.48145466, 0.4578275, 0.40821073)
            )
            model_info.config["dataset"].setdefault(
                "normalize_std", (0.26862954, 0.26130258, 0.27577711)
            )

        import logging, traceback

        logging.getLogger(__name__).info("HF model registered: %s", model_info.to_dict())

        # Persist to DB so it survives restarts
        try:
            from astra.app.database import get_db
            db = get_db()
            db.save_model_registration(
                model_id=model_info.model_id,
                architecture_path=model_info.architecture,
                config_json=json.dumps({
                    "source": "huggingface",
                    "model_type": model_info.model_type,
                    "use_peft": body.use_peft,
                    "peft_method": body.peft_method,
                    "total_params": model_info.total_params,
                    "trainable_params": model_info.trainable_params,
                }),
                is_huggingface=True,
            )
        except Exception:
            pass  # Don't fail the request if DB persist errors

        return {"status": "registered", "model": model_info.to_dict()}
    except Exception as e:
        import logging, traceback
        logging.getLogger(__name__).error("HF register failed: %s\n%s", e, traceback.format_exc())
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/api/models/register/architecture")
async def register_architecture(body: RegisterArchitectureBody):
    """Register an arbitrary PyTorch model architecture by import path.

    Accepts a dotted Python path like ``torchvision.models.resnet18``
    and registers it as a callable factory in the model registry.
    """
    fl_server = get_fl_server()

    try:
        module_path, attr_name = body.architecture_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        factory_fn = getattr(module, attr_name)
    except (ImportError, AttributeError) as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not import '{body.architecture_path}': {e}",
        ) from e

    if not callable(factory_fn):
        raise HTTPException(
            status_code=400,
            detail=f"'{body.architecture_path}' is not callable",
        )

    kwargs = body.config or {}
    try:
        model = factory_fn(**kwargs) if kwargs else factory_fn()
    except TypeError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to instantiate '{body.architecture_path}' with kwargs {kwargs}: {e}",
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to instantiate '{body.architecture_path}': {e}",
        ) from e

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    from astra.app.database import get_db
    from astra.infra.registry import ModelInfo

    if body.model_id in fl_server.model_registry.models:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{body.model_id}' is already registered",
        )
    try:
        db_rows = get_db().load_model_registrations()
        if any(r.get("model_id") == body.model_id for r in db_rows):
            raise HTTPException(
                status_code=400,
                detail=f"Model '{body.model_id}' is already registered",
            )
    except HTTPException:
        raise
    except Exception:
        pass  # DB lookup is best-effort

    model_info = ModelInfo(
        model_id=body.model_id,
        model_type=body.model_type,
        architecture=attr_name,
        total_params=total_params,
        trainable_params=trainable_params,
        is_peft=False,
        source="external",
        config={"architecture_path": body.architecture_path, "kwargs": kwargs},
    )

    fl_server.model_registry.register_factory(body.model_id, factory_fn, model_info)
    fl_server.model_registry.model_instances[body.model_id] = model

    # Persist to database for survival across restarts
    import json as _json

    get_db().save_model_registration(
        model_id=body.model_id,
        architecture=attr_name,
        architecture_path=body.architecture_path,
        config_json=_json.dumps({"kwargs": kwargs, "model_type": body.model_type}),
    )

    return {
        "status": "registered",
        "model": model_info.to_dict(),
    }

@router.get("/api/models/{model_id}")
async def get_model(model_id: str):
    """Get model details."""
    fl_server = get_fl_server()
    model_info = fl_server.model_registry.get_model_info(model_id)
    if not model_info:
        raise HTTPException(status_code=404, detail="Model not found")
    return {"model": model_info}


@router.get("/api/models/validate/{model_id}")
async def validate_model(model_id: str):
    """Validate model compatibility."""
    fl_server = get_fl_server()
    is_valid, message = fl_server.model_registry.validate_model(model_id)
    return {"model_id": model_id, "is_valid": is_valid, "message": message}


@router.get("/api/models/{group_id}/download")
async def download_model(
    group_id: str,
    version: int | None = None,
    format: str = "pt",
):
    """Download the global model weights for a group.

    If version is specified, downloads that version. Otherwise downloads latest.

    ``format``:
      - ``pt`` (default) — returns the PyTorch checkpoint file (.pt) as-is.
      - ``raw`` — loads the checkpoint, flattens the weights to a single
        little-endian float32 array, and returns the raw bytes. This is
        the format the ``/api/clients/{id}/delta`` and ``/api/uploads``
        endpoints expect.
    """
    fl_server = get_fl_server()
    if group_id not in fl_server.group_manager.groups:
        raise HTTPException(status_code=404, detail="Group not found")

    save_dir = os.path.join("models", "global", group_id)

    if version:
        file_path = os.path.join(save_dir, f"model_v{version}.pt")
    else:
        file_path = os.path.join(save_dir, "model_latest.pt")

    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=404, detail="Model file not found. No training has been completed yet."
        )

    if format == "pt":
        filename = f"{group_id}_model_v{version}.pt" if version else f"{group_id}_model_latest.pt"
        return FileResponse(file_path, media_type="application/octet-stream", filename=filename)

    if format == "raw":
        import numpy as np
        import torch

        ckpt = torch.load(file_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "weights" in ckpt:
            arr = ckpt["weights"]
        elif isinstance(ckpt, dict):
            # state_dict — concatenate values in deterministic order
            tensors = [v.detach().cpu().float().numpy().ravel() for v in ckpt.values()]
            arr = np.concatenate(tensors) if tensors else np.zeros(0, dtype="<f4")
        else:
            arr = ckpt.detach().cpu().float().numpy().ravel()

        raw_bytes = arr.astype("<f4").tobytes()
        filename = f"{group_id}_model_v{version}.bin" if version else f"{group_id}_model_latest.bin"
        return Response(
            content=raw_bytes,
            media_type="application/octet-stream",
            headers={
                "Content-Length": str(len(raw_bytes)),
                "X-Num-Parameters": str(arr.size),
                "X-Dtype": "<f4",
                "Content-Disposition": f'attachment; filename="{filename}"',
            },
        )

    raise HTTPException(status_code=400, detail=f"unknown format: {format!r}")


@router.get("/api/models/{group_id}/base")
async def download_base_model(group_id: str):
    """Download the frozen base model (non-LoRA backbone) for a group.

    Clients download this once and cache it locally. Only the LoRA adapter
    weights change across rounds.
    """
    fl_server = get_fl_server()
    if group_id not in fl_server.group_manager.groups:
        raise HTTPException(status_code=404, detail="Group not found")

    save_dir = os.path.join("models", "global", group_id)
    base_path = os.path.join(save_dir, "base.pt")

    if not os.path.exists(base_path):
        raise HTTPException(
            status_code=404,
            detail="Base model not found. Server must be started with PEFT enabled.",
        )

    return FileResponse(
        base_path,
        media_type="application/octet-stream",
        filename=f"{group_id}_base.pt",
    )


@router.get("/api/models/{group_id}/adapter")
async def download_latest_adapter(group_id: str):
    """Download the latest LoRA adapter weights for a group."""
    fl_server = get_fl_server()
    if group_id not in fl_server.group_manager.groups:
        raise HTTPException(status_code=404, detail="Group not found")

    save_dir = os.path.join("models", "global", group_id)
    adapter_path = os.path.join(save_dir, "adapter_latest.pt")

    if not os.path.exists(adapter_path):
        raise HTTPException(
            status_code=404,
            detail="No adapter weights available. No training has completed yet.",
        )

    return FileResponse(
        adapter_path,
        media_type="application/octet-stream",
        filename=f"{group_id}_adapter_latest.pt",
    )


@router.get("/api/models/{group_id}/adapter/{version}")
async def download_adapter_version(group_id: str, version: int):
    """Download a specific version of LoRA adapter weights."""
    fl_server = get_fl_server()
    if group_id not in fl_server.group_manager.groups:
        raise HTTPException(status_code=404, detail="Group not found")

    save_dir = os.path.join("models", "global", group_id)
    adapter_path = os.path.join(save_dir, f"adapter_v{version}.pt")

    if not os.path.exists(adapter_path):
        raise HTTPException(
            status_code=404,
            detail=f"Adapter version {version} not found.",
        )

    return FileResponse(
        adapter_path,
        media_type="application/octet-stream",
        filename=f"{group_id}_adapter_v{version}.pt",
    )


@router.get("/api/models/{group_id}/history")
async def get_model_history(group_id: str):
    """Get the full training history for a group.

    Returns model versions with accuracy, loss, timestamp, and number of
    contributing clients per round. Also returns the in-memory metrics.
    """
    fl_server = get_fl_server()
    if group_id not in fl_server.group_manager.groups:
        raise HTTPException(status_code=404, detail="Group not found")

    group = fl_server.group_manager.groups[group_id]

    # Get DB model records
    db = get_db()
    db_history = db.get_model_history(group_id, model_type="global")

    # Get in-memory metrics
    metrics = group.metrics_history

    # Check which model files exist on disk
    save_dir = os.path.join("models", "global", group_id)
    available_files = []
    if os.path.exists(save_dir):
        available_files = [f for f in os.listdir(save_dir) if f.endswith(".pt")]

    return {
        "group_id": group_id,
        "model_id": group.model_id,
        "current_version": group.model_version,
        "completed_rounds": group.completed_rounds,
        "history": db_history,
        "metrics": metrics,
        "available_files": available_files,
        "has_latest": os.path.exists(os.path.join(save_dir, "model_latest.pt"))
        if os.path.exists(save_dir)
        else False,
    }
