"""
Client management REST endpoints.
"""

import logging
from fastapi import APIRouter, HTTPException, Request

from networking.models import ClientRegister
from networking.state import get_fl_server

router = APIRouter()


@router.get("/api/clients/connected")
async def list_connected_clients():
    """List currently connected client IDs."""
    fl_server = get_fl_server()
    clients = list(fl_server.connection_manager.client_sockets.keys())
    return {"clients": clients, "count": len(clients)}


@router.post("/api/clients/register")
async def register_client(client: ClientRegister):
    """Register a client via REST."""
    fl_server = get_fl_server()
    client_id = client.client_id
    capabilities = client.capabilities

    # Register in database
    fl_server.db.register_fl_client(client_id, fl_server.experiment_id or 'default')

    # Add to connected clients
    fl_server.connection_manager.register_client(client_id, None)

    return {"status": "registered", "client_id": client_id}


@router.post("/api/join/activate/{group_id}")
async def join_group_as_client(group_id: str, request: Request):
    """Join an FL group as a participant after admin approval.

    Bridges the auth system (join request approval) with the FL system (group registration).
    The client must have an approved join request for this group.
    """
    fl_server = get_fl_server()

    # Verify JWT token
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="No authorization token")

    token = auth_header.replace("Bearer ", "")

    # Verify token via extended platform integration
    try:
        from api.integration import get_platform_integration
        platform = get_platform_integration()
        payload = platform.verify_token(token)
        if not payload:
            raise HTTPException(status_code=401, detail="Invalid token")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=401, detail="Token verification failed")

    user_id = payload.get("user_id")
    username = payload.get("sub", f"user_{user_id}")

    # Verify join request is approved
    try:
        status = platform.get_user_join_status(user_id, group_id)
        if not status or status.get("status") != "approved":
            raise HTTPException(
                status_code=403,
                detail="Join request not approved. Please request to join first and wait for admin approval."
            )
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=400, detail="Could not verify join status")

    # Check group exists in FL server
    group = fl_server.group_manager.groups.get(group_id)
    if not group:
        raise HTTPException(status_code=404, detail=f"Group '{group_id}' not found")

    # Generate a client ID from the username
    client_id = f"{username}_{group_id}"

    # Register client in the FL group
    group.add_client(client_id, {"user_id": user_id, "username": username})

    # Register in database
    fl_server.db.register_fl_client(client_id, fl_server.experiment_id or 'default')

    # Log the join event
    fl_server.group_manager.log_event(
        'client_joined',
        f'Client {username} joined group {group_id}',
        group_id,
        {'client_id': client_id, 'user_id': user_id, 'username': username}
    )

    # Auto-start training if this is the first client
    if len(group.clients) == 1 and not group.is_training:
        fl_server.group_manager.start_group_training(group_id)

    return {
        "status": "joined",
        "client_id": client_id,
        "group_id": group_id,
        "message": f"Successfully joined group {group_id}"
    }


@router.get("/api/clients")
async def list_clients():
    """List connected clients."""
    fl_server = get_fl_server()
    clients = fl_server.group_manager.get_all_client_status()
    logger = logging.getLogger(__name__)
    logger.info(f"[API-CLIENTS] Returning {len(clients)} clients")
    for c in clients:
        logger.info(f"  Client {c.get('client_id')} in group {c.get('group_id')}: acc={c.get('local_accuracy', 0):.4f}, loss={c.get('local_loss', 0):.4f}")
    return {"clients": clients, "count": len(clients)}


@router.get("/api/client/training-status")
async def get_client_training_status(request: Request):
    """Get training status for the authenticated client across all joined groups.

    Returns FL client entries matching the user's username, along with
    their group's training state, metrics, and model info.
    """
    fl_server = get_fl_server()

    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="No authorization token")

    token = auth_header.replace("Bearer ", "")

    try:
        from api.integration import get_platform_integration
        platform = get_platform_integration()
        payload = platform.verify_token(token)
        if not payload:
            raise HTTPException(status_code=401, detail="Invalid token")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=401, detail="Token verification failed")

    username = payload.get("sub", "")
    user_id = payload.get("user_id")

    # Find all FL clients belonging to this user (pattern: {username}_{group_id})
    sessions = []
    for group_id, group in fl_server.group_manager.groups.items():
        for client_id, client_info in group.clients.items():
            # Match by username prefix or by user_id in client_info
            is_match = (
                client_id.startswith(f"{username}_") or
                client_info.get("user_id") == user_id or
                client_info.get("username") == username
            )
            if is_match:
                # Get the group's latest metrics
                latest_metrics = {}
                if group.metrics_history:
                    last = group.metrics_history[-1]
                    latest_metrics = {
                        "global_accuracy": last.get("accuracy", 0),
                        "global_loss": last.get("loss", 0),
                        "global_version": last.get("version", 0),
                    }

                sessions.append({
                    "client_id": client_id,
                    "group_id": group_id,
                    "model_id": group.model_id,
                    "group_status": group.status,
                    "is_training": group.is_training,
                    "local_accuracy": client_info.get("local_accuracy", 0),
                    "local_loss": client_info.get("local_loss", 0),
                    "trust_score": client_info.get("trust_score", 1.0),
                    "updates_count": client_info.get("updates_count", 0),
                    "last_update": client_info.get("last_update"),
                    "status": client_info.get("status", "idle"),
                    "joined_at": client_info.get("joined_at"),
                    "model_version": group.model_version,
                    "window_status": group.get_window_status(),
                    **latest_metrics,
                })

    # Also check which groups user has approved join requests for but hasn't activated yet
    pending_activations = []
    try:
        from api.integration import get_platform_integration
        platform = get_platform_integration()
        # Get all groups and check join status
        for group_id in fl_server.group_manager.groups:
            already_joined = any(s["group_id"] == group_id for s in sessions)
            if not already_joined:
                try:
                    status = platform.get_user_join_status(user_id, group_id)
                    if status and status.get("status") == "approved":
                        pending_activations.append({
                            "group_id": group_id,
                            "model_id": fl_server.group_manager.groups[group_id].model_id,
                            "status": "approved_not_activated",
                        })
                except Exception:
                    pass
    except Exception:
        pass

    # Check if any WebSocket client is connected for this user
    connected_ws_clients = []
    for client_id in fl_server.connection_manager.client_sockets:
        if client_id.startswith(f"{username}_"):
            connected_ws_clients.append(client_id)

    return {
        "username": username,
        "sessions": sessions,
        "pending_activations": pending_activations,
        "connected_clients": connected_ws_clients,
        "has_active_training": any(s["is_training"] for s in sessions),
    }
