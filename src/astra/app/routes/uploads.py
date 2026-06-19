"""
Upload REST endpoints for the presigned-URL flow.

Endpoints:
    POST /api/uploads/init
        Body: { client_id, group_id, content_length, sha256 }
        Auth: Bearer JWT
        Returns: { upload_id, upload_url, expires_at, content_length }
        403 if the client_id in the body doesn't match the JWT
        413 if content_length exceeds the global max
        507 if disk is full

    PUT /api/uploads/{upload_id}/blob?expires=...&sig=...
        Body: raw bytes (chunked or one-shot)
        Auth: HMAC signature in `sig` query param (no JWT)
        Returns: 200 with { received, total } once the upload equals
                 declared size, OR 200 with { received, total } mid-flight
                 to allow resumability.

    POST /api/uploads/{upload_id}/complete
        Body: { sha256 }
        Auth: Bearer JWT (must match the client_id from init)
        Returns: 200 with { status: "completed", sha256, size } on success
                 Triggers the dispatch into the FLServer pipeline.

    GET /api/uploads/{upload_id}
        Auth: Bearer JWT (must match the client_id)
        Returns: current UploadRecord

    DELETE /api/uploads/{upload_id}
        Auth: Bearer JWT (must match the client_id)
        Aborts the upload and frees disk.
"""

import logging
from collections import deque

import numpy as np
from fastapi import APIRouter, Header, HTTPException, Request

from astra.app.routes._auth import verify_request_jwt
from astra.app.state import get_fl_server
from astra.app.uploads import get_upload_manager

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/api/uploads/init")
async def init_upload(request: Request, authorization: str = Header(None)):
    """Allocate a new upload slot. Returns a presigned PUT URL."""
    payload = verify_request_jwt(authorization)
    user_id = payload.get("user_id")
    if not isinstance(user_id, int):
        raise HTTPException(status_code=401, detail="Invalid user_id in token")

    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON body: {e}") from None

    client_id = body.get("client_id")
    group_id = body.get("group_id")
    content_length = body.get("content_length")
    sha256 = body.get("sha256")

    if not client_id:
        raise HTTPException(status_code=400, detail="client_id is required")
    if not isinstance(content_length, int) or content_length <= 0:
        raise HTTPException(
            status_code=400, detail="content_length must be a positive integer"
        )
    if not sha256 or not isinstance(sha256, str) or len(sha256) != 64:
        raise HTTPException(status_code=400, detail="sha256 must be a 64-char hex string")

    try:
        manager = get_upload_manager()
        record, _placeholder, expires_at = manager.init_upload(
            client_id=client_id,
            group_id=group_id,
            declared_size=content_length,
            declared_sha256=sha256.lower(),
        )
    except RuntimeError as e:
        raise HTTPException(status_code=507, detail=str(e)) from None
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None

    upload_url, _ = manager.sign_put_url(record.upload_id)
    from astra.core.config import DEFAULT_CONFIG as _DC
    uploads_cfg = _DC.get("uploads", {})
    return {
        "upload_id": record.upload_id,
        "upload_url": upload_url,
        "expires_at": record.expires_at,
        "content_length": record.declared_size,
        "expires_in_seconds": int(record.expires_at - record.created_at),
        "chunk_size": uploads_cfg.get("chunk_size", 8 * 1024 * 1024),
        "max_inline_bytes": uploads_cfg.get("max_inline_bytes", 100 * 1024 * 1024),
    }


@router.put("/api/uploads/{upload_id}/blob")
async def upload_blob(
    upload_id: str,
    request: Request,
    expires: float,
    sig: str,
):
    """PUT the delta bytes to the presigned URL.

    Supports:
      - single PUT with the full body
      - resumable PUT with Content-Range-style offsets via Content-Length

    On each call, the bytes received so far are appended to the staged
    file. When the staged file reaches the declared size, the upload is
    auto-finalized and a 200 response indicates readiness for /complete.
    """
    manager = get_upload_manager()
    if not manager.verify(upload_id, "put", expires, sig):
        raise HTTPException(status_code=403, detail="invalid or expired signature")

    record = manager.get(upload_id)
    if record is None:
        raise HTTPException(status_code=404, detail="upload not found")
    if record.status in ("completed", "aborted", "failed"):
        raise HTTPException(
            status_code=409, detail=f"upload already in state {record.status}"
        )

    # Read the entire body. FastAPI streams under the hood, but for
    # correctness with sha256 verification we want all bytes.
    body_bytes = await request.body()
    if not body_bytes:
        raise HTTPException(status_code=400, detail="empty body")

    if record.status == "initiated":
        manager.start_receiving(upload_id)

    new_received = manager.append(upload_id, body_bytes)
    return {
        "upload_id": upload_id,
        "received": new_received,
        "total": record.declared_size,
        "complete": new_received >= record.declared_size,
    }


@router.post("/api/uploads/{upload_id}/complete")
async def complete_upload(upload_id: str, request: Request, authorization: str = Header(None)):
    """Verify sha256 + dispatch the staged delta into the FLServer pipeline."""
    verify_request_jwt(authorization)
    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON body: {e}") from None

    expected_sha = (body.get("sha256") or "").lower() or None

    manager = get_upload_manager()
    try:
        record = manager.complete(upload_id, expected_sha256=expected_sha)
    except KeyError:
        raise HTTPException(status_code=404, detail="upload not found") from None
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None
    except RuntimeError as e:
        raise HTTPException(status_code=410, detail=str(e)) from None

    # Find the staged blob and dispatch through the existing delta path.
    blob_path = manager.store._blob_path(upload_id)
    delta_bytes = blob_path.read_bytes()

    # Reuse the validation + dispatch logic from /api/clients/{id}/delta
    return await _dispatch_staged_delta(record, delta_bytes)


@router.get("/api/uploads/{upload_id}")
async def get_upload(upload_id: str, authorization: str = Header(None)):
    """Read the upload record. Any authenticated user can read any upload.

    (Authorization is currently coarse — there's no per-upload ACL.
    The presigned PUT URL acts as a capability for the upload itself,
    so the only privilege an attacker gains by reading the record is
    metadata leakage. Tighten if that becomes a concern.)
    """
    verify_request_jwt(authorization)
    manager = get_upload_manager()
    record = manager.get(upload_id)
    if record is None:
        raise HTTPException(status_code=404, detail="upload not found")
    return record.to_dict()


@router.delete("/api/uploads/{upload_id}")
async def abort_upload(upload_id: str, authorization: str = Header(None)):
    """Abort an in-progress upload. Same auth model as GET."""
    verify_request_jwt(authorization)
    manager = get_upload_manager()
    record = manager.get(upload_id)
    if record is None:
        raise HTTPException(status_code=404, detail="upload not found")
    manager.abort(upload_id)
    return {"status": "aborted", "upload_id": upload_id}


async def _dispatch_staged_delta(record, delta_bytes: bytes) -> dict:
    """Validate the staged bytes and push them into the AsyncServer.

    Mirrors the validation+dispatch logic from /api/clients/{id}/delta
    but reads bytes from disk instead of the request body.
    """
    fl_server = get_fl_server()

    client_id = record.client_id

    # Look up group + model
    group = fl_server.group_manager.get_client_group(client_id)
    if not group:
        raise HTTPException(
            status_code=404, detail=f"client {client_id} not registered in any group"
        )

    # Param-count-based size validation (same as inline path)
    from astra.app.routes.clients import _get_expected_param_count

    expected_params = _get_expected_param_count(fl_server, group.model_id)
    if expected_params is not None:
        expected_f32 = expected_params * 4
        expected_f64 = expected_params * 8
        if len(delta_bytes) not in (expected_f32, expected_f64):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"delta byte length ({len(delta_bytes):,}) does not match the "
                    f"expected size for model '{group.model_id}' "
                    f"({expected_params:,} parameters). Expected {expected_f32:,} "
                    f"bytes (float32) or {expected_f64:,} bytes (float64). Did you "
                    f"upload a PyTorch checkpoint instead of raw float32 weight bytes?"
                ),
            )

    # PEFT validation: if the group uses PEFT, the delta should be adapter
    # weights only (much smaller than full model).
    is_peft = group.config.get("peft", {}).get("enabled", False)
    if is_peft and expected_params is not None:
        full_model_bytes = expected_params * 4
        adapter_ratio = len(delta_bytes) / full_model_bytes if full_model_bytes > 0 else 1.0
        if adapter_ratio > 0.5:
            logger.warning(
                "Client %s in PEFT group %s uploaded %.1f%% of full model size "
                "(%d bytes vs %d expected for full model) via presigned URL flow.",
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

    delta = np.frombuffer(delta_bytes, dtype="<f4")
    if not np.all(np.isfinite(delta)):
        raise HTTPException(status_code=400, detail="delta contains NaN or Inf values")

    if not group.is_training:
        return {"status": "rejected", "reason": "group_not_training", "global_version": 0}

    # Sync per-group config into the AsyncServer so its aggregator window
    # matches the group's window_size (independent of when the FLServer was
    # initialized — important when tests share a process and the FLServer was
    # created with default config).
    fl_server.config.setdefault("model", {})["model_id"] = group.model_id
    fl_server.config.setdefault("server", {})["aggregator_window"] = (
        group.window_config.window_size
    )

    # Lazy AsyncServer init (only first time the FLServer is used)
    if fl_server.server is None:
        try:
            fl_server._setup_server()
        except Exception as e:
            logger.exception("Lazy server init failed")
            return {"status": "rejected", "reason": f"server_init_failed: {e}"}

    # Push the updated aggregator_window into the live AsyncServer in case it
    # was already running with the old default.
    if fl_server.server is not None and fl_server.server.config.get(
        "server", {}
    ).get("aggregator_window") != group.window_config.window_size:
        fl_server.server.config.setdefault("server", {})[
            "aggregator_window"
        ] = group.window_config.window_size
        fl_server.server.aggregator_buffer = deque(
            maxlen=group.window_config.window_size
        )

    if fl_server.server is None:
        return {"status": "rejected", "reason": "server_not_ready"}

    client_update = {
        "client_id": client_id,
        "client_version": 0,
        "local_updates": delta.tobytes(),
        "update_type": "delta",
        "local_dataset_size": 1,
        "timestamp": 0,
        "meta": {},
    }
    fl_server.server.handle_update(client_update)
    new_version = fl_server.server.global_version

    triggered = fl_server.group_manager.process_client_update(client_id, client_update)
    if triggered.get("aggregate"):
        fl_server.group_manager.aggregate_group(group.group_id)

    # Best-effort: free disk once dispatched
    import contextlib as _cl
    with _cl.suppress(Exception):
        manager = get_upload_manager()
        manager.store.discard(record.upload_id)

    return {
        "status": "completed",
        "sha256": record.actual_sha256,
        "size": record.declared_size,
        "global_version": new_version,
    }
