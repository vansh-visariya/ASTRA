"""
Client management REST endpoints.

Clients train externally and submit pre-computed deltas via
POST /api/clients/{client_id}/delta. The server aggregates
them with the configured aggregator and broadcasts the
new global model on the WebSocket.
"""

import base64
import contextlib
import logging
import os
import time

from fastapi import APIRouter, HTTPException, Request

from astra.app.state import get_fl_server
from astra.infra.models import ClientRegister, ClientUpdate

logger = logging.getLogger(__name__)
router = APIRouter()

MAX_DELTA_BYTES = int(
    os.environ.get("ASTRA_DELTA_CAP_BYTES", str(100 * 1024 * 1024))
)  # default 100 MB cap on decoded delta
MIN_DELTA_INTERVAL_S = 2.0  # per-client rate limit

_last_delta_at: dict[str, float] = {}


def _get_expected_param_count(fl_server, model_id: str) -> int | None:
    """Look up the registered model's total parameter count.

    Returns None if the model can't be built (missing import, broken
    factory, etc.) — callers should fall back to a lenient check.
    """
    try:
        model = fl_server.model_registry.build_model(model_id)
        return sum(p.numel() for p in model.parameters())
    except Exception:
        return None


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

    fl_server.db.register_fl_client(client_id, fl_server.experiment_id or "default")
    fl_server.connection_manager.register_client(client_id, None)  # type: ignore[arg-type]

    return {"status": "registered", "client_id": client_id}


@router.post("/api/clients/{client_id}/delta")
async def submit_client_delta(client_id: str, request: Request):
    """Submit a pre-computed model delta from an external training process.

    Body: ClientUpdate (client_version, local_updates as base64 float32 bytes,
    local_dataset_size, optional meta). Requires a valid JWT.
    """
    fl_server = get_fl_server()

    # Auth
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="No authorization token")
    token = auth_header.replace("Bearer ", "")
    try:
        from astra.app.integration import get_platform_integration

        platform = get_platform_integration()
        payload = platform.verify_token(token)
        if not payload:
            raise HTTPException(status_code=401, detail="Invalid token")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=401, detail="Token verification failed") from None

    # Rate limit
    now = time.monotonic()
    last = _last_delta_at.get(client_id, 0.0)
    if now - last < MIN_DELTA_INTERVAL_S:
        wait = MIN_DELTA_INTERVAL_S - (now - last)
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit: wait {wait:.1f}s before next upload",
        )
    _last_delta_at[client_id] = now

    # Parse body
    try:
        body = await request.json()
        update = ClientUpdate(**body)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid request body: {e}") from None

    if update.client_id != client_id:
        raise HTTPException(
            status_code=400,
            detail=f"client_id in body ({update.client_id}) does not match URL ({client_id})",
        )

    # Decode base64
    try:
        delta_bytes = base64.b64decode(update.local_updates, validate=True)
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"local_updates is not valid base64: {e}"
        ) from None

    # Large-payload handoff: if the inline delta exceeds the configured cap,
    # tell the client to use the presigned-URL upload flow instead.
    if len(delta_bytes) > MAX_DELTA_BYTES:
        from astra.app.uploads import get_upload_manager as _get_um

        manager = _get_um()
        # We can't compute sha256 here (the client didn't send it), so we
        # require the client to call /api/uploads/init themselves.
        raise HTTPException(
            status_code=413,
            detail=(
                f"delta payload too large for inline upload "
                f"({len(delta_bytes):,} bytes; inline cap is "
                f"{MAX_DELTA_BYTES // 1024 // 1024} MB). Use the presigned-URL "
                f"flow: POST /api/uploads/init with content_length and "
                f"sha256, then PUT the raw bytes to the returned upload_url, "
                f"then POST /api/uploads/{{upload_id}}/complete."
            ),
            headers={
                "X-Astra-Upload-Cap-Bytes": str(MAX_DELTA_BYTES),
                "X-Astra-Upload-Init-Path": "/api/uploads/init",
                "X-Astra-Upload-Manager-Ready": "1" if manager else "0",
            },
        )

    # Find the group this client belongs to — we need its model_id to
    # validate the delta size against the expected parameter count.
    group = fl_server.group_manager.get_client_group(client_id)
    if not group:
        raise HTTPException(
            status_code=404,
            detail=f"client {client_id} is not registered in any group",
        )

    # Compute expected delta size from the registered model's param count.
    # The delta must be exactly num_parameters * 4 bytes (float32) OR
    # num_parameters * 8 bytes (float64). Anything else is the wrong file
    # (e.g. a PyTorch checkpoint, a state_dict pickle, a .npy with header).
    expected_params = _get_expected_param_count(fl_server, group.model_id)
    if expected_params is None:
        # Model can't be built (missing import, broken factory). Allow the
        # size check to pass and let the dispatch path fail later with a
        # better diagnostic. This keeps legitimate uploads flowing even
        # when the registry has a broken entry.
        expected_params = None
        expected_f32_bytes = None
        expected_f64_bytes = None
    else:
        expected_f32_bytes = expected_params * 4
        expected_f64_bytes = expected_params * 8

    # Size validation with actionable error message. Note: the early
    # handoff above already converted any payload > MAX_DELTA_BYTES into
    # a 413 with upload-flow instructions, so reaching this branch means
    # the payload is within the inline cap but the model param count
    # itself is bigger than the cap (i.e. the model simply can't be
    # uploaded via this endpoint at all).
    if len(delta_bytes) > MAX_DELTA_BYTES:
        if expected_f32_bytes is not None and expected_f32_bytes > MAX_DELTA_BYTES:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"delta payload too large ({len(delta_bytes)} bytes). "
                    f"This model has {expected_params:,} parameters; the "
                    f"expected float32 delta is {expected_f32_bytes:,} bytes "
                    f"({expected_f32_bytes / 1024 / 1024:.1f} MB). The inline "
                    f"upload cap is {MAX_DELTA_BYTES // 1024 // 1024} MB. "
                    f"This model is too large to upload via this endpoint. "
                    f"Raise MAX_DELTA_BYTES in config.yaml or run the server "
                    f"with a custom ASTRA_DELTA_CAP_BYTES env var."
                ),
            )
        raise HTTPException(
            status_code=413,
            detail=f"delta payload too large ({len(delta_bytes)} bytes); max is {MAX_DELTA_BYTES}",
        )

    # Check PEFT from group config OR model registry
    is_peft = group.config.get("peft", {}).get("enabled", False)
    if not is_peft:
        model_info = fl_server.model_registry.get_model_info(group.model_id)
        if model_info:
            is_peft = model_info.get("is_peft", False)

    if expected_f32_bytes is not None and not is_peft:
        # Strict size check for non-PEFT models only.
        # PEFT groups upload adapter-only deltas (much smaller than full model).
        if len(delta_bytes) not in (expected_f32_bytes, expected_f64_bytes):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"delta byte length ({len(delta_bytes):,}) does not match "
                    f"the expected size for model '{group.model_id}' "
                    f"({expected_params:,} parameters). "
                    f"Expected {expected_f32_bytes:,} bytes (float32) or "
                    f"{expected_f64_bytes:,} bytes (float64). "
                    f"Did you upload a PyTorch checkpoint (.pt from "
                    f"/api/models/{{group_id}}/download) instead of raw "
                    f"weight bytes? The download returns a checkpoint "
                    f"dictionary — extract the 'weights' array, flatten it, "
                    f"cast to float32 (.astype('<f4')), and call .tobytes()."
                ),
            )
    elif not is_peft:
        # Model param count unknown — fall back to the lenient % 4 check.
        if len(delta_bytes) % 4 != 0:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"delta byte length ({len(delta_bytes)}) is not a multiple "
                    f"of 4 (float32). The model registry has no param count "
                    f"for '{group.model_id}'; cannot validate size. Did you "
                    f"upload a serialized state_dict or PyTorch checkpoint?"
                ),
            )

    # For PEFT groups, at least check % 4 (must be float32 bytes)
    if is_peft and len(delta_bytes) % 4 != 0:
        raise HTTPException(
            status_code=400,
            detail=f"adapter delta byte length ({len(delta_bytes)}) is not a multiple of 4 (float32)",
        )

    # Validate payload before dispatching — NaN/Inf always rejected.
    import numpy as np

    # Use explicit little-endian dtype — np.frombuffer with dtype=np.float32 on
    # big-endian numpy builds (some Windows configurations) reads bytes in
    # network order, silently producing garbled deltas.
    delta = np.frombuffer(delta_bytes, dtype="<f4")
    if not np.all(np.isfinite(delta)):
        raise HTTPException(status_code=400, detail="delta contains NaN or Inf values")

    # PEFT validation: if the group uses PEFT, the delta should be adapter
    # weights only (much smaller than full model). Warn if the client
    # uploaded full model weights by mistake.
    if is_peft and expected_params is not None:
        # For PEFT, expected_params is the full model param count.
        # The adapter delta should be a fraction of that.
        # Typical LoRA rank=8 on q_proj/v_proj is ~0.1-2% of total params.
        full_model_bytes = expected_params * 4
        adapter_ratio = len(delta_bytes) / full_model_bytes if full_model_bytes > 0 else 1.0

        if adapter_ratio > 0.5:
            # Client likely uploaded full model weights instead of adapter delta
            logger.warning(
                "Client %s in PEFT group %s uploaded %.1f%% of full model size "
                "(%d bytes vs %d expected for full model). This is likely full "
                "model weights, not an adapter delta.",
                client_id,
                group.group_id,
                adapter_ratio * 100,
                len(delta_bytes),
                full_model_bytes,
            )
            raise HTTPException(
                status_code=400,
                detail=(
                    f"PEFT group '{group.group_id}' expects adapter-only delta "
                    f"(LoRA weights), but the uploaded payload is "
                    f"{adapter_ratio * 100:.1f}% of the full model size "
                    f"({len(delta_bytes):,} bytes vs {full_model_bytes:,} for full model). "
                    f"Download the base model once, fine-tune locally, then upload "
                    f"only the adapter weights. Use flatten_peft_params() from "
                    f"astra.core.models.model_zoo to extract adapter parameters."
                ),
            )

    # Dispatch to FL server pipeline
    if fl_server.is_paused:
        return {"status": "rejected", "reason": "server_paused"}

    if not group.is_training:
        return {"status": "rejected", "reason": "group_not_training"}

    # If the global AsyncServer isn't built yet (e.g., server started with
    # no model_id), build it lazily using the group's model_id and window.
    if fl_server.server is None:
        try:
            if not fl_server.config.get("model", {}).get("model_id"):
                fl_server.config.setdefault("model", {})["model_id"] = group.model_id
            fl_server.config.setdefault("server", {})["aggregator_window"] = (
                group.window_config.window_size
            )
            fl_server._setup_server()
        except Exception as e:
            logger.exception("Lazy server init failed")
            return {"status": "rejected", "reason": f"server_init_failed: {e}"}

    if fl_server.server is None:
        return {"status": "rejected", "reason": "server_not_ready"}

    client_update = {
        "client_id": update.client_id,
        "client_version": update.client_version,
        "local_updates": delta.tobytes(),
        "update_type": update.update_type,
        "local_dataset_size": update.local_dataset_size,
        "timestamp": time.time(),
        "meta": update.meta,
    }

    # Dispatch via the AsyncServer (applies DP if configured, updates trust,
    # and triggers aggregation when the window fills).
    fl_server.server.handle_update(client_update)
    new_version = fl_server.server.global_version

    # Auto-mark server as running once it has accepted at least one update
    if not fl_server.is_running:
        fl_server.is_running = True

    # Also let the GroupManager aggregate so the group-level model_version
    # and metrics stay in sync with the AsyncServer's global_version.
    # The aggregator is the same one used by the WebSocket path.
    triggered = fl_server.group_manager.process_client_update(
        client_id, client_update
    )
    if triggered.get("aggregate"):
        fl_server.group_manager.aggregate_group(group.group_id)

    # Bump client's update counter / last_update
    try:
        fl_server.db.update_fl_client_metrics(
            client_id=client_id,
            local_accuracy=update.meta.get("train_accuracy", 0.0),
            local_loss=update.meta.get("train_loss", 0.0),
            updates_count=group.clients.get(client_id, {}).get("updates_count", 0),
            status="active",
        )
    except Exception as e:
        logger.warning("Could not persist client metrics for %s: %s", client_id, e)

    # Broadcast to dashboard (best-effort)
    with contextlib.suppress(Exception):
        await fl_server.connection_manager.broadcast(
            {
                "type": "client_update",
                "client_id": client_id,
                "step": new_version,
            }
        )

    return {
        "status": "accepted",
        "client_id": client_id,
        "global_version": new_version,
    }


@router.post("/api/join/activate/{group_id}")
async def join_group_as_client(group_id: str, request: Request):
    """Join an FL group as a participant after admin approval.

    The user is registered as an FL client in the group. They can then
    train externally and submit deltas via POST /api/clients/{client_id}/delta.
    """
    fl_server = get_fl_server()

    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="No authorization token")

    token = auth_header.replace("Bearer ", "")

    try:
        from astra.app.integration import get_platform_integration

        platform = get_platform_integration()
        payload = platform.verify_token(token)
        if not payload:
            raise HTTPException(status_code=401, detail="Invalid token")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=401, detail="Token verification failed") from None

    user_id = payload.get("user_id")
    if not isinstance(user_id, int):
        raise HTTPException(status_code=401, detail="Invalid user_id in token")
    username = payload.get("sub", f"user_{user_id}")

    try:
        status = platform.get_user_join_status(user_id, group_id)
        if not status or status.get("status") != "approved":
            raise HTTPException(
                status_code=403,
                detail=(
                    "Join request not approved. Please request to join"
                    " first and wait for admin approval."
                ),
            )
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=400, detail="Could not verify join status") from None

    group = fl_server.group_manager.groups.get(group_id)
    if not group:
        raise HTTPException(status_code=404, detail=f"Group '{group_id}' not found")

    client_id = f"{username}_{group_id}"

    # Register client via GroupManager.register_client so client_to_group is updated
    success = fl_server.group_manager.register_client(
        client_id=client_id,
        group_id=group_id,
        client_info={"user_id": user_id, "username": username},
    )
    if not success:
        raise HTTPException(status_code=400, detail="Could not register client in group")

    try:
        platform.auth_manager.join_request_manager.mark_user_activated(user_id, group_id)
    except Exception as e:
        logger.warning("Failed to mark user %s activated in group %s: %s", user_id, group_id, e)

    fl_server.group_manager.log_event(
        "client_joined",
        f"Client {username} joined group {group_id}",
        group_id,
        {"client_id": client_id, "user_id": user_id, "username": username},
    )

    if group.is_training:
        with contextlib.suppress(RuntimeError):
            await fl_server.group_manager.notify_training_started(group_id)

    return {
        "status": "joined",
        "client_id": client_id,
        "group_id": group_id,
        "message": f"Successfully joined group {group_id}",
    }


@router.get("/api/clients")
async def list_clients():
    """List all known FL clients across groups."""
    fl_server = get_fl_server()
    clients = fl_server.group_manager.get_all_client_status()
    return {"clients": clients, "count": len(clients)}


@router.get("/api/clients/{client_id}/status")
async def get_client_status(client_id: str, request: Request):
    """Return the latest server-known status for a single client.

    Includes the current global model version, the client's last-update
    timestamp, and the global accuracy/loss from the most recent aggregation
    in the client's group. Requires authentication.
    """
    fl_server = get_fl_server()

    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="No authorization token")
    token = auth_header.replace("Bearer ", "")
    try:
        from astra.app.integration import get_platform_integration

        platform = get_platform_integration()
        payload = platform.verify_token(token)
        if not payload:
            raise HTTPException(status_code=401, detail="Invalid token")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=401, detail="Token verification failed") from None

    group = fl_server.group_manager.get_client_group(client_id)
    if not group:
        raise HTTPException(status_code=404, detail=f"Client {client_id} not found")

    latest = group.metrics_history[-1] if group.metrics_history else {}
    client_info = group.clients.get(client_id, {})

    return {
        "client_id": client_id,
        "group_id": group.group_id,
        "model_id": group.model_id,
        "is_training": group.is_training,
        "global_version": group.model_version,
        "global_accuracy": latest.get("accuracy", 0),
        "global_loss": latest.get("loss", 0),
        "last_update": client_info.get("last_update"),
        "updates_count": client_info.get("updates_count", 0),
        "trust_score": client_info.get("trust_score", 1.0),
    }
