"""
Local-disk object store + upload manager for the presigned-URL upload flow.

Background: clients need to upload multi-GB deltas. A single POST that
size over WAN fails 90% of the time (timeouts, drops, no resumability).
The fix is to let the client PUT the bytes directly to a presigned URL
in 8 MB chunks, with sha256 verification on completion.

Flow:
    Client                   Server                          Disk
      |  POST /api/uploads/init --->|                              |
      |  {size, sha256}            +-- generate HMAC-signed URL -->|
      <-- {upload_id, put_url} ----+                              |
      |                            |                              |
      |  PUT put_url (chunked) ---+---------------->-------------->|
      |  (with progress events)    |                              |
      |                            |                              |
      |  POST /api/uploads/{id}/complete ->|                      |
      |  {sha256}                  +-- verify sha256, dispatch ---|
      <-- {status, global_version} -+                              |

Storage layout (LocalDiskObjectStore):
    ./uploads/
        <upload_id>.bin        # raw bytes
        <upload_id>.meta.json  # {client_id, group_id, size, sha256,
                                #  presigned_at, status, expires_at}
        <upload_id>.lock       # simple flock marker while PUT in progress
"""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import json
import os
import secrets
import shutil
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from astra.core.config import DEFAULT_CONFIG


@dataclass
class UploadRecord:
    """Persistent metadata for a single staged upload."""

    upload_id: str
    client_id: str
    group_id: str | None
    declared_size: int
    declared_sha256: str
    created_at: float
    expires_at: float
    status: str = "initiated"  # initiated | receiving | verifying | completed | aborted | failed
    received_bytes: int = 0
    actual_sha256: str | None = None
    completed_at: float | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UploadRecord:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


class LocalDiskObjectStore:
    """Filesystem-backed object store for staged uploads."""

    def __init__(self, disk_path: str, min_free_bytes: int = 1024 * 1024 * 1024):
        self.disk_path = Path(disk_path)
        self.disk_path.mkdir(parents=True, exist_ok=True)
        self.min_free_bytes = min_free_bytes
        self._locks: dict[str, threading.RLock] = {}
        self._locks_lock = threading.Lock()

    def _lock_for(self, upload_id: str) -> threading.RLock:
        with self._locks_lock:
            lock = self._locks.get(upload_id)
            if lock is None:
                lock = threading.RLock()
                self._locks[upload_id] = lock
            return lock

    def _check_disk_space(self, need_bytes: int) -> bool:
        try:
            usage = shutil.disk_usage(self.disk_path)
            return usage.free - need_bytes > self.min_free_bytes
        except OSError:
            return False

    def _meta_path(self, upload_id: str) -> Path:
        return self.disk_path / f"{upload_id}.meta.json"

    def _blob_path(self, upload_id: str) -> Path:
        return self.disk_path / f"{upload_id}.bin"

    def _tmp_path(self, upload_id: str) -> Path:
        return self.disk_path / f"{upload_id}.bin.tmp"

    def write_meta(self, record: UploadRecord) -> None:
        with self._lock_for(record.upload_id):
            meta_path = self._meta_path(record.upload_id)
            tmp = meta_path.with_name(meta_path.name + ".tmp")
            # On Windows os.replace can fail with WinError 5 if the target
            # still has handles held by the filesystem cache.  Unlink first
            # to avoid the race.
            if meta_path.exists():
                with contextlib.suppress(OSError):
                    meta_path.unlink()
            tmp.write_text(json.dumps(record.to_dict()))
            os.replace(tmp, meta_path)

    def read_meta(self, upload_id: str) -> UploadRecord | None:
        path = self._meta_path(upload_id)
        if not path.exists():
            return None
        try:
            return UploadRecord.from_dict(json.loads(path.read_text()))
        except Exception:
            return None

    def start_receive(self, upload_id: str) -> None:
        with self._lock_for(upload_id):
            tmp = self._tmp_path(upload_id)
            if tmp.exists():
                tmp.unlink()
            tmp.touch()

    def append_chunk(self, upload_id: str, chunk: bytes) -> int:
        with self._lock_for(upload_id):
            with open(self._tmp_path(upload_id), "ab") as f:
                f.write(chunk)
            return self._tmp_path(upload_id).stat().st_size

    def finalize(self, upload_id: str) -> tuple[Path, int, str]:
        with self._lock_for(upload_id):
            tmp = self._tmp_path(upload_id)
            blob = self._blob_path(upload_id)
            if blob.exists():
                blob.unlink()
            os.replace(tmp, blob)
            size = blob.stat().st_size
            sha = hashlib.sha256(blob.read_bytes()).hexdigest()
            return blob, size, sha

    def discard(self, upload_id: str) -> None:
        with self._lock_for(upload_id):
            for p in (
                self._tmp_path(upload_id),
                self._blob_path(upload_id),
                self._meta_path(upload_id),
            ):
                if p.exists():
                    with contextlib.suppress(OSError):
                        p.unlink()

    def cleanup_expired(self, max_age_seconds: int) -> int:
        cutoff = time.time() - max_age_seconds
        removed = 0
        for meta_file in self.disk_path.glob("*.meta.json"):
            try:
                data = json.loads(meta_file.read_text())
                completed_at = data.get("completed_at")
                created_at = data.get("created_at")
                is_expired_completed = bool(completed_at and completed_at < cutoff)
                is_stale_unfinished = bool(
                    created_at
                    and created_at < cutoff
                    and data.get("status") != "completed"
                )
                if is_expired_completed or is_stale_unfinished:
                    self.discard(data["upload_id"])
                    removed += 1
            except Exception:
                continue
        return removed

    def free_disk_bytes(self) -> int:
        try:
            return shutil.disk_usage(self.disk_path).free
        except OSError:
            return 0


class UploadManager:
    """Coordinates the lifecycle of staged uploads."""

    def __init__(
        self,
        store: LocalDiskObjectStore,
        secret_key: bytes,
        presign_ttl: int = 3600,
    ):
        self.store = store
        self.secret_key = secret_key
        self.presign_ttl = presign_ttl
        self._records: dict[str, UploadRecord] = {}
        self._lock = threading.Lock()
        self._load_existing()

    def _load_existing(self) -> None:
        for meta_file in self.store.disk_path.glob("*.meta.json"):
            try:
                data = json.loads(meta_file.read_text())
                rec = UploadRecord.from_dict(data)
                if rec.status != "completed" and rec.expires_at < time.time():
                    self.store.discard(rec.upload_id)
                    continue
                self._records[rec.upload_id] = rec
            except Exception:
                continue

    @staticmethod
    def _new_id() -> str:
        return secrets.token_urlsafe(24)

    def _sign(self, upload_id: str, action: str, expires_at: float) -> str:
        msg = f"{upload_id}:{action}:{int(expires_at)}".encode()
        return hmac.new(self.secret_key, msg, hashlib.sha256).hexdigest()

    def verify(
        self, upload_id: str, action: str, expires_at: float, signature: str
    ) -> bool:
        if expires_at < time.time():
            return False
        expected = self._sign(upload_id, action, expires_at)
        return hmac.compare_digest(expected, signature)

    def init_upload(
        self,
        client_id: str,
        group_id: str | None,
        declared_size: int,
        declared_sha256: str,
    ) -> tuple[UploadRecord, str, float]:
        if declared_size <= 0:
            raise ValueError("declared_size must be positive")
        if not self.store._check_disk_space(declared_size):
            raise RuntimeError("insufficient disk space")

        upload_id = self._new_id()
        now = time.time()
        expires_at = now + self.presign_ttl
        record = UploadRecord(
            upload_id=upload_id,
            client_id=client_id,
            group_id=group_id,
            declared_size=declared_size,
            declared_sha256=declared_sha256,
            created_at=now,
            expires_at=expires_at,
        )
        with self._lock:
            self._records[upload_id] = record
        self.store.write_meta(record)
        put_path = (
            f"/api/uploads/{upload_id}/blob?expires={int(expires_at)}&sig=PLACEHOLDER"
        )
        return record, put_path, expires_at

    def sign_put_url(self, upload_id: str) -> tuple[str, float]:
        rec = self._records.get(upload_id)
        if rec is None:
            raise KeyError(upload_id)
        sig = self._sign(upload_id, "put", rec.expires_at)
        path = f"/api/uploads/{upload_id}/blob?expires={int(rec.expires_at)}&sig={sig}"
        return path, rec.expires_at

    def start_receiving(self, upload_id: str) -> None:
        rec = self._require(upload_id)
        if rec.status not in ("initiated", "receiving"):
            raise RuntimeError(
                f"upload {upload_id} is in state {rec.status}, cannot receive"
            )
        self.store.start_receive(upload_id)
        rec.status = "receiving"
        self.store.write_meta(rec)

    def append(self, upload_id: str, chunk: bytes) -> int:
        rec = self._require(upload_id)
        if rec.status not in ("receiving",):
            raise RuntimeError(f"upload {upload_id} is not in receiving state")
        new_size = self.store.append_chunk(upload_id, chunk)
        rec.received_bytes = new_size
        self.store.write_meta(rec)
        return new_size

    def complete(
        self, upload_id: str, expected_sha256: str | None = None
    ) -> UploadRecord:
        rec = self._require(upload_id)
        if rec.status not in ("receiving",):
            raise RuntimeError(
                f"upload {upload_id} is in state {rec.status}, cannot complete"
            )
        rec.status = "verifying"
        self.store.write_meta(rec)
        try:
            blob_path, actual_size, actual_sha = self.store.finalize(upload_id)
        except Exception as e:
            rec.status = "failed"
            rec.error = f"finalize failed: {e}"
            self.store.write_meta(rec)
            raise
        if actual_size != rec.declared_size:
            rec.status = "failed"
            rec.error = f"size mismatch: declared {rec.declared_size}, got {actual_size}"
            self.store.write_meta(rec)
            raise ValueError(rec.error)
        if expected_sha256 and not hmac.compare_digest(
            actual_sha, expected_sha256.lower()
        ):
            rec.status = "failed"
            rec.error = "sha256 mismatch"
            self.store.write_meta(rec)
            raise ValueError(rec.error)
        if not hmac.compare_digest(actual_sha, rec.declared_sha256.lower()):
            rec.status = "failed"
            rec.error = (
                f"sha256 mismatch: declared {rec.declared_sha256}, got {actual_sha}"
            )
            self.store.write_meta(rec)
            raise ValueError(rec.error)
        rec.actual_sha256 = actual_sha
        rec.status = "completed"
        rec.completed_at = time.time()
        self.store.write_meta(rec)
        return rec

    def abort(self, upload_id: str) -> None:
        rec = self._records.get(upload_id)
        if rec is not None:
            rec.status = "aborted"
            self.store.write_meta(rec)
        self.store.discard(upload_id)
        with self._lock:
            self._records.pop(upload_id, None)

    def get(self, upload_id: str) -> UploadRecord | None:
        return self._records.get(upload_id)

    def list_for_client(self, client_id: str) -> list[UploadRecord]:
        return [r for r in self._records.values() if r.client_id == client_id]

    def _require(self, upload_id: str) -> UploadRecord:
        rec = self._records.get(upload_id)
        if rec is None:
            raise KeyError(upload_id)
        if rec.expires_at < time.time() and rec.status != "completed":
            raise RuntimeError(f"upload {upload_id} has expired")
        return rec


# ----- Module-level singleton wired by the lifespan -----


_upload_manager: UploadManager | None = None


def init_upload_manager(
    config: dict | None = None,
    secret_key: bytes | None = None,
) -> UploadManager:
    """Build the singleton upload manager from config + secret key."""
    global _upload_manager
    cfg = config or DEFAULT_CONFIG.get("uploads", {})
    store = LocalDiskObjectStore(
        disk_path=cfg.get("disk_path", "./uploads"),
        min_free_bytes=cfg.get("min_free_disk_bytes", 1024 * 1024 * 1024),
    )
    key = secret_key or os.environ.get("SECRET_KEY", "astra-dev-secret").encode()
    _upload_manager = UploadManager(
        store=store,
        secret_key=key,
        presign_ttl=cfg.get("presign_ttl_seconds", 3600),
    )
    return _upload_manager


def get_upload_manager() -> UploadManager:
    if _upload_manager is None:
        raise RuntimeError(
            "Upload manager not initialized. The application lifespan must "
            "call init_upload_manager() during startup."
        )
    return _upload_manager
