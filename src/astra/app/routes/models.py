"""
Model management REST endpoints.
"""

import importlib
import json
import os
from typing import Any

import requests
from fastapi import APIRouter, HTTPException, Query
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
                architecture=model_info.architecture,
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
      - ``safetensors`` — loads the checkpoint and returns weights in
        safetensors format (for HuggingFace-native clients).
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

    if format == "safetensors":
        import torch

        try:
            from safetensors.torch import save_file as _safetensors_save
        except ImportError:
            raise HTTPException(
                status_code=400,
                detail="safetensors package not installed on server",
            ) from None

        ckpt = torch.load(file_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "weights" in ckpt:
            # Flat numpy array — can't directly save as safetensors
            # Fall back to returning raw bytes with a warning
            raise HTTPException(
                status_code=400,
                detail=(
                    "This checkpoint is stored as a flat array and cannot be "
                    "converted to safetensors. Use format=raw instead."
                ),
            )
        elif isinstance(ckpt, dict):
            # state_dict — filter to tensors only
            tensor_dict = {}
            for k, v in ckpt.items():
                if isinstance(v, torch.Tensor):
                    tensor_dict[k] = v.cpu()
            if not tensor_dict:
                raise HTTPException(
                    status_code=400, detail="No tensors found in checkpoint"
                )
        else:
            raise HTTPException(
                status_code=400,
                detail="Checkpoint format not compatible with safetensors",
            )

        # Save to a temp file and return it
        tmp_path = os.path.join(save_dir, f"_tmp_safetensors_v{version or 'latest'}.safetensors")
        try:
            _safetensors_save(tensor_dict, tmp_path)
            filename = (
                f"{group_id}_model_v{version}.safetensors"
                if version
                else f"{group_id}_model_latest.safetensors"
            )
            return FileResponse(
                tmp_path,
                media_type="application/octet-stream",
                filename=filename,
                background=None,
            )
        finally:
            # Clean up temp file after response is sent
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    raise HTTPException(status_code=400, detail=f"unknown format: {format!r}")


@router.get("/api/models/{group_id}/base")
async def download_base_model(
    group_id: str,
    format: str = Query("pt", pattern="^(pt|safetensors)$"),
):
    """Download the frozen base model (non-LoRA backbone) for a group.

    Clients download this once and cache it locally. Only the LoRA adapter
    weights change across rounds.

    ``format``:
      - ``pt`` (default) — PyTorch .pt file.
      - ``safetensors`` — safetensors format (for HuggingFace-native clients).
    """
    fl_server = get_fl_server()
    if group_id not in fl_server.group_manager.groups:
        raise HTTPException(status_code=404, detail="Group not found")

    group = fl_server.group_manager.groups[group_id]

    # Try models/hf/{model_id}/ first (saved during group creation)
    hf_dir = os.path.join("models", "hf", group.model_id)
    # Fallback to models/global/{group_id}/
    global_dir = os.path.join("models", "global", group_id)

    # For safetensors format, prefer the HF directory
    if format == "safetensors":
        sf_path = os.path.join(hf_dir, "base_model.safetensors")
        if os.path.exists(sf_path):
            return FileResponse(
                sf_path,
                media_type="application/octet-stream",
                filename=f"{group_id}_base.safetensors",
            )
        # Try global dir
        sf_path_global = os.path.join(global_dir, "base_model.safetensors")
        if os.path.exists(sf_path_global):
            return FileResponse(
                sf_path_global,
                media_type="application/octet-stream",
                filename=f"{group_id}_base.safetensors",
            )
        # If no safetensors file exists, try to convert from .pt
        pt_path = os.path.join(hf_dir, "base_model.pt")
        if not os.path.exists(pt_path):
            pt_path = os.path.join(global_dir, "base.pt")
        if os.path.exists(pt_path):
            try:
                from safetensors.torch import save_file as _safetensors_save

                data = torch.load(pt_path, map_location="cpu", weights_only=False)
                base_state = data.get("base_state_dict", data)
                if isinstance(base_state, dict):
                    tensor_dict = {
                        k: v.cpu() for k, v in base_state.items()
                        if isinstance(v, torch.Tensor)
                    }
                    if tensor_dict:
                        tmp_path = os.path.join(hf_dir, "_tmp_base.safetensors")
                        os.makedirs(hf_dir, exist_ok=True)
                        _safetensors_save(tensor_dict, tmp_path)
                        return FileResponse(
                            tmp_path,
                            media_type="application/octet-stream",
                            filename=f"{group_id}_base.safetensors",
                        )
            except ImportError:
                raise HTTPException(
                    status_code=400,
                    detail="safetensors package not installed on server and no safetensors file available",
                ) from None
        raise HTTPException(
            status_code=404,
            detail="Base model not found. Server must be started with PEFT enabled.",
        )

    # .pt format (default)
    pt_path = os.path.join(hf_dir, "base_model.pt")
    if not os.path.exists(pt_path):
        pt_path = os.path.join(global_dir, "base.pt")
    if not os.path.exists(pt_path):
        raise HTTPException(
            status_code=404,
            detail="Base model not found. Server must be started with PEFT enabled.",
        )

    return FileResponse(
        pt_path,
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


@router.get("/api/models/{group_id}/download-info")
async def get_download_info(group_id: str):
    """Get metadata about available model files for a group.

    Returns info about base model, adapter, and available formats.
    Clients use this to decide what to download (base vs adapter, pt vs safetensors).
    """
    fl_server = get_fl_server()
    if group_id not in fl_server.group_manager.groups:
        raise HTTPException(status_code=404, detail="Group not found")

    group = fl_server.group_manager.groups[group_id]

    # Check PEFT from group config OR model registry
    is_peft = group.config.get("peft", {}).get("enabled", False)
    if not is_peft:
        # Also check the model registry — PEFT info lives there when
        # the group was created via the HuggingFace tab
        model_info = fl_server.model_registry.get_model_info(group.model_id)
        if model_info:
            is_peft = model_info.get("is_peft", False)

    # Check HF directory
    hf_dir = os.path.join("models", "hf", group.model_id)
    global_dir = os.path.join("models", "global", group_id)

    # Get base model info
    from astra.core.models.hf_models import get_download_info as _get_hf_info

    hf_info = _get_hf_info(hf_dir)

    # Get global model info (adapter + versioned checkpoints)
    global_info: dict[str, Any] = {"has_adapter": False, "adapter_versions": []}
    adapter_latest = os.path.join(global_dir, "adapter_latest.pt")
    if os.path.exists(adapter_latest):
        global_info["has_adapter"] = True
        global_info["adapter_latest_size"] = os.path.getsize(adapter_latest)

    if os.path.exists(global_dir):
        for fname in os.listdir(global_dir):
            if fname.startswith("adapter_v") and fname.endswith(".pt"):
                try:
                    ver = int(fname.replace("adapter_v", "").replace(".pt", ""))
                    global_info["adapter_versions"].append(ver)
                except ValueError:
                    pass
    global_info["adapter_versions"].sort()

    # Check for latest global model
    latest_path = os.path.join(global_dir, "model_latest.pt")
    has_global_model = os.path.exists(latest_path)

    return {
        "group_id": group_id,
        "model_id": group.model_id,
        "is_peft": is_peft,
        "base_model": {
            "available": hf_info.get("has_base_model", False),
            "formats": list(hf_info.get("formats", {}).keys()),
            "sizes": {
                fmt: info.get("size_bytes", 0)
                for fmt, info in hf_info.get("formats", {}).items()
            },
        },
        "adapter": {
            "available": global_info.get("has_adapter", False),
            "versions": global_info.get("adapter_versions", []),
            "latest_size": global_info.get("adapter_latest_size", 0),
        },
        "global_model": {
            "available": has_global_model,
            "current_version": group.model_version,
        },
    }
