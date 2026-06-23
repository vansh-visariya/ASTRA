"""
Group management REST endpoints.
"""

from fastapi import APIRouter, Depends, Header, HTTPException

from astra.app.integration import get_platform_integration
from astra.app.state import get_fl_server
from astra.infra.models import TrainingManifest

router = APIRouter()


def _get_any_user(authorization: str = Header(None)):
    """Require valid JWT token (any role)."""
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization required")
    token = authorization.replace("Bearer ", "")
    platform = get_platform_integration()
    payload = platform.verify_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid token")
    return payload


def _require_admin(authorization: str = Header(None)):
    """Require valid JWT token with admin role."""
    payload = _get_any_user(authorization)
    if payload.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return payload


@router.get("/api/groups")
async def list_groups(current_user=Depends(_get_any_user)):
    """List all training groups with their async window status (authenticated users only)."""
    fl_server = get_fl_server()
    groups = fl_server.group_manager.get_all_groups(include_secret=False)
    return {"groups": groups, "count": len(groups)}


@router.post("/api/groups")
async def create_group(group_data: dict, current_user=Depends(_get_any_user)):
    """Create a new training group."""
    fl_server = get_fl_server()
    group_id = group_data.get("group_id")
    if not isinstance(group_id, str):
        raise HTTPException(status_code=400, detail="group_id is required and must be a string")
    model_id = group_data.get("model_id")
    if not model_id:
        raise HTTPException(status_code=400, detail="model_id is required")
    window_size = group_data.get("window_size", 3)
    if not isinstance(window_size, int) or window_size < 1:
        raise HTTPException(
            status_code=400,
            detail=f"window_size must be a positive integer (got {window_size!r})",
        )
    time_limit = group_data.get("time_limit", 20.0)
    if not isinstance(time_limit, (int, float)) or time_limit <= 0:
        raise HTTPException(
            status_code=400,
            detail=f"time_limit must be a positive number (got {time_limit!r})",
        )
    custom_token = group_data.get("join_token")

    # Build config with training parameters
    aggregator_name = group_data.get("aggregator", "fedavg")
    config = {
        "join_token": custom_token if custom_token else "GENERATE_NEW",
        "local_epochs": group_data.get("local_epochs", 2),
        "batch_size": group_data.get("batch_size", 32),
        "lr": group_data.get("lr", 0.01),
        "aggregator": aggregator_name,
        "dp_enabled": group_data.get("dp_enabled", False),
    }

    # Map the flat aggregator name to the nested robust.method key that
    # create_aggregator() reads.  Without this, the UI dropdown has no
    # effect on actual aggregation behavior.
    if aggregator_name not in ("fedavg", ""):
        config.setdefault("robust", {})["method"] = aggregator_name

    # --- Training Manifest (optional but recommended) ---
    manifest_data = group_data.get("training_manifest")
    manifest = None
    if manifest_data:
        try:
            manifest = TrainingManifest(**manifest_data)
            # Store manifest in group config for persistence
            config["training_manifest"] = manifest.model_dump()
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid training_manifest: {e}",
            )

    group = fl_server.group_manager.create_group(
        group_id=group_id,
        model_id=model_id,
        config=config,
        window_size=window_size,
        time_limit=time_limit,
        created_by=current_user.get("user_id"),
    )

    # For create we still return the real token once, for the admin caller.
    result = group.to_dict(include_secret=True)
    return {"status": "created", "group": result}


@router.get("/api/groups/{group_id}")
async def get_group(group_id: str, current_user=Depends(_get_any_user)):
    """Get specific group details. Join token only visible to admins."""
    fl_server = get_fl_server()
    group = fl_server.group_manager.groups.get(group_id)
    if not group:
        raise HTTPException(status_code=404, detail="Group not found")
    is_admin = current_user.get("role") == "admin"
    return {"group": group.to_dict(include_secret=is_admin)}


@router.get("/api/groups/{group_id}/manifest")
async def get_group_manifest(group_id: str):
    """Get the training manifest for a group.

    Clients call this before training to learn the required architecture,
    training protocol, and data schema.
    """
    fl_server = get_fl_server()
    group = fl_server.group_manager.groups.get(group_id)
    if not group:
        raise HTTPException(status_code=404, detail="Group not found")

    manifest = group.config.get("training_manifest")
    if not manifest:
        raise HTTPException(
            status_code=404,
            detail="No training manifest defined for this group. "
            "The admin should create the group with a training_manifest.",
        )
    return {
        "group_id": group_id,
        "model_id": group.model_id,
        "manifest": manifest,
    }


@router.post("/api/groups/{group_id}/start")
async def start_group_training(group_id: str, current_user=Depends(_require_admin)):
    """Start accepting deltas for a group (admin only)."""
    fl_server = get_fl_server()
    success = fl_server.group_manager.start_group_training(group_id)
    if not success:
        raise HTTPException(status_code=400, detail="Cannot start training")

    await fl_server.group_manager.notify_training_started(group_id)

    return {"status": "started", "group_id": group_id}


@router.post("/api/groups/{group_id}/pause")
async def pause_group_training(group_id: str, current_user=Depends(_require_admin)):
    """Pause accepting deltas for a group (admin only)."""
    fl_server = get_fl_server()
    success = fl_server.group_manager.pause_group_training(group_id)
    if not success:
        raise HTTPException(status_code=400, detail="Cannot pause training")
    return {"status": "paused", "group_id": group_id}


@router.post("/api/groups/{group_id}/resume")
async def resume_group_training(group_id: str, current_user=Depends(_require_admin)):
    """Resume accepting deltas for a group (admin only)."""
    fl_server = get_fl_server()
    success = fl_server.group_manager.resume_group_training(group_id)
    if not success:
        raise HTTPException(status_code=400, detail="Cannot resume training")
    await fl_server.group_manager.notify_training_started(group_id)
    return {"status": "resumed", "group_id": group_id}


@router.post("/api/groups/{group_id}/stop")
async def stop_group_training(group_id: str, current_user=Depends(_require_admin)):
    """Stop accepting deltas for a group (admin only)."""
    fl_server = get_fl_server()
    success = fl_server.group_manager.stop_group_training(group_id)
    if not success:
        raise HTTPException(status_code=400, detail="Cannot stop training")
    return {"status": "stopped", "group_id": group_id}


@router.delete("/api/groups/{group_id}")
async def delete_group(group_id: str, current_user=Depends(_require_admin)):
    """Delete a training group and all associated data (admin only)."""
    fl_server = get_fl_server()
    success = fl_server.group_manager.delete_group(group_id)
    if not success:
        raise HTTPException(status_code=404, detail="Group not found or could not be deleted")
    return {"status": "deleted", "group_id": group_id}


@router.get("/api/groups/{group_id}/window-status")
async def get_group_window_status(group_id: str):
    """Get async window status for a group."""
    fl_server = get_fl_server()
    group = fl_server.group_manager.groups.get(group_id)
    if not group:
        raise HTTPException(status_code=404, detail="Group not found")
    return {
        "group_id": group_id,
        "status": group.status,
        "is_training": group.is_training,
        "model_version": group.model_version,
        "window_status": group.get_window_status(),
    }
