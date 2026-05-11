"""
Group management REST endpoints.
"""

from typing import Dict

from fastapi import APIRouter, HTTPException

from networking.state import get_fl_server

router = APIRouter()


@router.get("/api/groups")
async def list_groups():
    """List all training groups with their async window status."""
    fl_server = get_fl_server()
    # Do NOT expose raw join tokens in the general listing.
    groups = fl_server.group_manager.get_all_groups(include_secret=False)
    return {"groups": groups, "count": len(groups)}


@router.post("/api/groups")
async def create_group(group_data: Dict):
    """Create a new training group."""
    fl_server = get_fl_server()
    group_id = group_data.get('group_id')
    model_id = group_data.get('model_id', 'simple_cnn_mnist')
    window_size = group_data.get('window_size', 3)
    time_limit = group_data.get('time_limit', 20.0)
    custom_token = group_data.get('join_token')

    # Build config with training parameters
    config = {
        'join_token': custom_token if custom_token else "GENERATE_NEW",
        'local_epochs': group_data.get('local_epochs', 2),
        'batch_size': group_data.get('batch_size', 32),
        'lr': group_data.get('lr', 0.01),
        'aggregator': group_data.get('aggregator', 'fedavg'),
        'dp_enabled': group_data.get('dp_enabled', False),
    }

    group = fl_server.group_manager.create_group(
        group_id=group_id,
        model_id=model_id,
        config=config,
        window_size=window_size,
        time_limit=time_limit
    )

    # For create we still return the real token once, for the admin caller.
    result = group.to_dict(include_secret=True)
    return {"status": "created", "group": result}


@router.get("/api/groups/{group_id}")
async def get_group(group_id: str):
    """Get specific group details (admin view with token)."""
    fl_server = get_fl_server()
    group = fl_server.group_manager.groups.get(group_id)
    if not group:
        raise HTTPException(status_code=404, detail="Group not found")
    return {"group": group.to_dict(include_secret=True)}


@router.post("/api/groups/{group_id}/start")
async def start_group_training(group_id: str):
    """Start training for a group - clients train independently."""
    fl_server = get_fl_server()
    success = fl_server.group_manager.start_group_training(group_id)
    if not success:
        raise HTTPException(status_code=400, detail="Cannot start training")

    # Notify all clients that training is now open - they train autonomously
    await fl_server.group_manager.notify_training_started(group_id)

    return {"status": "started", "group_id": group_id}


@router.post("/api/groups/{group_id}/pause")
async def pause_group_training(group_id: str):
    """Pause training for a group."""
    fl_server = get_fl_server()
    success = fl_server.group_manager.pause_group_training(group_id)
    if not success:
        raise HTTPException(status_code=400, detail="Cannot pause training")
    await fl_server.group_manager.notify_training_paused(group_id)
    return {"status": "paused", "group_id": group_id}


@router.post("/api/groups/{group_id}/resume")
async def resume_group_training(group_id: str):
    """Resume training for a group."""
    fl_server = get_fl_server()
    success = fl_server.group_manager.resume_group_training(group_id)
    if not success:
        raise HTTPException(status_code=400, detail="Cannot resume training")
    await fl_server.group_manager.notify_training_started(group_id)
    return {"status": "resumed", "group_id": group_id}


@router.post("/api/groups/{group_id}/stop")
async def stop_group_training(group_id: str):
    """Stop training for a group."""
    fl_server = get_fl_server()
    success = fl_server.group_manager.stop_group_training(group_id)
    if not success:
        raise HTTPException(status_code=400, detail="Cannot stop training")
    await fl_server.group_manager.notify_training_stopped(group_id)
    return {"status": "stopped", "group_id": group_id}


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
        "window_status": group.get_window_status()
    }
