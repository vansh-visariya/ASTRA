"""
Chunked, resumable model-download endpoints.

Mirrors the upload flow in `routes/uploads.py` but in the opposite
direction:

  init → list of signed chunk URLs → GET each chunk → complete (telemetry)

The single-shot `/api/models/{group_id}/download?format=...` endpoint in
`routes/models.py` still works for small files. Large files should use
this flow.
"""

import hashlib
import logging
import time
from pathlib import Path

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import Response

from astra.app.downloads import get_download_manager
from astra.app.routes._auth import verify_request_jwt
from astra.app.state import get_fl_server
from astra.core.config import DEFAULT_CONFIG

logger = logging.getLogger(__name__)
router = APIRouter()


def _resolve_model_file(group_id: str, version: int | None, fmt: str, download_type: str | None = None) -> Path:
    """Return the on-disk path to the model file for this group/version/format/type."""
    fl_server = get_fl_server()
    if group_id not in fl_server.group_manager.groups:
        raise HTTPException(status_code=404, detail="Group not found")

    group = fl_server.group_manager.groups[group_id]

    # For base model downloads (PEFT groups)
    if download_type == "base":
        hf_dir = Path("models") / "hf" / group.model_id
        global_dir = Path("models") / "global" / group_id

        pt_path = hf_dir / "base_model.pt"
        if pt_path.exists():
            return pt_path
        pt_path = global_dir / "base.pt"
        if pt_path.exists():
            return pt_path
        raise HTTPException(
            status_code=404,
            detail="Base model not found. The group may not have PEFT enabled.",
        )

    # For adapter downloads (PEFT groups)
    if download_type == "adapter":
        save_dir = Path("models") / "global" / group_id
        adapter_path = save_dir / "adapter_latest.pt"
        if adapter_path.exists():
            return adapter_path
        raise HTTPException(
            status_code=404,
            detail="No adapter weights available. No training has completed yet.",
        )

    # Standard model download (global model)
    save_dir = Path("models") / "global" / group_id
    if not save_dir.exists():
        raise HTTPException(status_code=404, detail="Group has no saved models yet")
    p = save_dir / f"model_v{version}.pt" if version is not None else save_dir / "model_latest.pt"
    if not p.exists():
        raise HTTPException(
            status_code=404, detail="No model file for that version"
        )
    return p


@router.post("/api/downloads/init")
async def init_download(request: Request, authorization: str = Header(None)):
    """Allocate a chunked download slot. Returns a manifest with signed URLs."""
    payload = verify_request_jwt(authorization)
    user_id = payload.get("user_id")
    if not isinstance(user_id, int):
        raise HTTPException(status_code=401, detail="Invalid user_id in token")

    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON body: {e}") from None

    group_id = body.get("group_id")
    version = body.get("version")
    fmt = body.get("format", "pt")
    download_type = body.get("download_type")  # "base" or "adapter" for PEFT groups
    if not group_id:
        raise HTTPException(status_code=400, detail="group_id is required")
    if fmt not in ("pt", "raw"):
        raise HTTPException(status_code=400, detail="format must be 'pt' or 'raw'")
    if version is not None and not isinstance(version, int):
        raise HTTPException(status_code=400, detail="version must be an integer")

    source_path = _resolve_model_file(group_id, version, fmt, download_type=download_type)

    uploads_cfg = DEFAULT_CONFIG.get("uploads", {})
    chunk_size = int(uploads_cfg.get("chunk_size", 8 * 1024 * 1024))
    if "chunk_size" in body and isinstance(body["chunk_size"], int):
        # If the client requests a smaller chunk size (e.g. for tiny test
        # files or large-model fine-grained progress), honor it. Only clamp
        # the upper bound to the server default so a single client can't
        # request arbitrarily large chunks.
        requested = body["chunk_size"]
        chunk_size = min(chunk_size, requested) if requested > 0 else chunk_size

    try:
        manager = get_download_manager()
        rec = manager.init_download(
            user_id=user_id,
            group_id=group_id,
            version=version,
            format=fmt,
            source_path=str(source_path),
            chunk_size=chunk_size,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from None

    chunk_urls = []
    for i in range(rec.num_chunks):
        url, _expires = manager.sign_chunk_url(rec.download_id, i)
        chunk_urls.append({"index": i, "url": url})

    return {
        "download_id": rec.download_id,
        "group_id": rec.group_id,
        "version": rec.version,
        "format": rec.format,
        "total_size": rec.total_size,
        "sha256": rec.sha256,
        "chunk_size": rec.chunk_size,
        "num_chunks": rec.num_chunks,
        "expires_at": rec.expires_at,
        "expires_in_seconds": int(rec.expires_at - rec.created_at),
        "chunks": chunk_urls,
    }


@router.get("/api/downloads/{download_id}/chunk/{chunk_index}")
async def get_chunk(download_id: str, chunk_index: int, expires: float, sig: str):
    """Stream one chunk from the staged model file. Signed URL only."""
    manager = get_download_manager()
    if not manager.verify(download_id, chunk_index, expires, sig):
        raise HTTPException(status_code=403, detail="invalid or expired signature")

    rec = manager.get(download_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="download not found")
    if rec.expires_at < time.time() and rec.bytes_served < rec.total_size:
        raise HTTPException(status_code=410, detail="download has expired")
    if chunk_index < 0 or chunk_index >= rec.num_chunks:
        raise HTTPException(status_code=404, detail="chunk_index out of range")

    start = chunk_index * rec.chunk_size
    end = min(start + rec.chunk_size, rec.total_size)
    length = end - start

    path = Path(rec.source_path)
    if not path.exists():
        raise HTTPException(status_code=410, detail="source file no longer exists")

    with open(path, "rb") as f:
        f.seek(start)
        data = f.read(length)

    manager.record_chunk_served(download_id, len(data))
    return Response(
        content=data,
        media_type="application/octet-stream",
        headers={
            "Content-Length": str(length),
            "Content-Range": f"bytes {start}-{end - 1}/{rec.total_size}",
            "X-Chunk-Index": str(chunk_index),
            "X-Num-Chunks": str(rec.num_chunks),
            "X-Chunk-Sha256": hashlib.sha256(data).hexdigest(),
            "X-Download-Id": download_id,
            "X-Group-Id": rec.group_id,
            "X-Total-Sha256": rec.sha256,
        },
    )


@router.post("/api/downloads/{download_id}/complete")
async def complete_download(download_id: str, authorization: str = Header(None)):
    """Mark a download as finished. Purely for server-side telemetry."""
    payload = verify_request_jwt(authorization)
    user_id = payload.get("user_id")
    manager = get_download_manager()
    rec = manager.get(download_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="download not found")
    if rec.user_id != user_id:
        raise HTTPException(status_code=403, detail="download belongs to another user")
    manager.mark_complete(download_id)
    return {
        "download_id": download_id,
        "status": "completed",
        "bytes_served": rec.bytes_served,
        "total_size": rec.total_size,
        "sha256": rec.sha256,
    }


@router.delete("/api/downloads/{download_id}")
async def abort_download(download_id: str, authorization: str = Header(None)):
    """Abort a download and free its slot."""
    payload = verify_request_jwt(authorization)
    user_id = payload.get("user_id")
    manager = get_download_manager()
    rec = manager.get(download_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="download not found")
    if rec.user_id != user_id:
        raise HTTPException(status_code=403, detail="download belongs to another user")
    manager.abort(download_id)
    return {"download_id": download_id, "status": "aborted"}


@router.get("/api/downloads/{download_id}")
async def get_download_info(download_id: str, authorization: str = Header(None)):
    """Inspect a download slot's state."""
    payload = verify_request_jwt(authorization)
    user_id = payload.get("user_id")
    manager = get_download_manager()
    rec = manager.get(download_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="download not found")
    if rec.user_id != user_id:
        raise HTTPException(status_code=403, detail="download belongs to another user")
    return rec.to_dict()
