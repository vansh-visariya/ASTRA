"""
Model management REST endpoints.
"""

import os
from typing import Any, Dict

import requests
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from api.database import get_db
from networking.state import get_fl_server

router = APIRouter()


@router.post("/api/models/register")
async def register_model(model_id: str, model_type: str, model_source: str, config: Dict):
    """Register a new model."""
    fl_server = get_fl_server()
    if model_source == "huggingface":
        model_info = fl_server.model_registry.register_hf_model(
            config['model_name'],
            use_peft=config.get('use_peft', False),
            peft_config=config.get('peft_config')
        )
    elif model_source == "custom":
        model_info = fl_server.model_registry.register_custom_architecture(
            model_id,
            config['architecture'],
            model_type,
            config
        )

    return {"status": "registered", "model": model_info.to_dict()}


@router.get("/api/models")
async def list_models():
    """List all available models."""
    fl_server = get_fl_server()
    models = fl_server.model_registry.list_models()
    return {"models": models, "count": len(models)}


def _fetch_hf_model_metadata(model_name: str) -> Dict[str, Any]:
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
async def register_hf_model(model_name: str, use_peft: bool = False, peft_method: str = "lora"):
    """Register a HuggingFace model."""
    fl_server = get_fl_server()
    try:
        peft_config = {
            'enabled': use_peft,
            'method': peft_method,
            'lora_rank': 8,
            'lora_alpha': 16,
            'target_modules': ['q_proj', 'v_proj']
        } if use_peft else {'enabled': False}

        model_info = fl_server.model_registry.register_hf_model(
            model_name=model_name,
            use_peft=use_peft,
            peft_config=peft_config
        )

        hf_meta = _fetch_hf_model_metadata(model_name)
        hf_config = (hf_meta.get('config') or {})
        vision_config = hf_config.get('vision_config') or {}
        image_size = hf_config.get('image_size') or vision_config.get('image_size')
        if image_size:
            model_info.config.setdefault('dataset', {})
            model_info.config['dataset'].setdefault('image_size', image_size)
            model_info.config['dataset'].setdefault('channels', 3)
            model_info.config['dataset'].setdefault(
                'normalize_mean',
                (0.48145466, 0.4578275, 0.40821073)
            )
            model_info.config['dataset'].setdefault(
                'normalize_std',
                (0.26862954, 0.26130258, 0.27577711)
            )

        return {"status": "registered", "model": model_info.to_dict()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


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
async def download_model(group_id: str, version: int = None):
    """Download the global model weights for a group.

    If version is specified, downloads that version. Otherwise downloads latest.
    Returns the .pt file as a binary download.
    """
    fl_server = get_fl_server()
    if group_id not in fl_server.group_manager.groups:
        raise HTTPException(status_code=404, detail="Group not found")

    save_dir = os.path.join('models', 'global', group_id)

    if version:
        file_path = os.path.join(save_dir, f'model_v{version}.pt')
    else:
        file_path = os.path.join(save_dir, 'model_latest.pt')

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Model file not found. No training has been completed yet.")

    filename = f"{group_id}_model_v{version}.pt" if version else f"{group_id}_model_latest.pt"
    return FileResponse(
        file_path,
        media_type="application/octet-stream",
        filename=filename
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
    db_history = db.get_model_history(group_id, model_type='global')

    # Get in-memory metrics
    metrics = group.metrics_history

    # Check which model files exist on disk
    save_dir = os.path.join('models', 'global', group_id)
    available_files = []
    if os.path.exists(save_dir):
        available_files = [f for f in os.listdir(save_dir) if f.endswith('.pt')]

    return {
        "group_id": group_id,
        "model_id": group.model_id,
        "current_version": group.model_version,
        "completed_rounds": group.completed_rounds,
        "history": db_history,
        "metrics": metrics,
        "available_files": available_files,
        "has_latest": os.path.exists(os.path.join(save_dir, 'model_latest.pt')) if os.path.exists(save_dir) else False,
    }
