"""
DownloadManager — chunked, resumable, sha256-verified downloads of large
model files. Symmetric to the UploadManager in `astra.app.uploads`.

Flow:
  client → POST /api/downloads/init  → {download_id, manifest}
  client → GET  /api/downloads/{id}/chunk/{N}?expires=&sig=  → bytes
  client → POST /api/downloads/{id}/complete (telemetry only)

The chunk URLs are short-lived HMAC-signed so a leaked URL doesn't leak
the whole model forever.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from astra.core.config import DEFAULT_CONFIG


@dataclass
class DownloadRecord:
    download_id: str
    user_id: int
    group_id: str
    version: int | None
    format: str  # "pt" or "raw"
    source_path: str
    total_size: int
    sha256: str
    chunk_size: int
    num_chunks: int
    created_at: float
    expires_at: float
    completed_at: float | None = None
    bytes_served: int = 0
    status: str = "ready"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DownloadManager:
    """Allocates chunked download slots and signs per-chunk GET URLs."""

    def __init__(self, secret_key: bytes, presign_ttl: int = 3600):
        self.secret_key = secret_key
        self.presign_ttl = presign_ttl
        self._records: dict[str, DownloadRecord] = {}
        self._lock = threading.Lock()

    def init_download(
        self,
        user_id: int,
        group_id: str,
        version: int | None,
        format: str,
        source_path: str,
        chunk_size: int,
    ) -> DownloadRecord:
        """Create a download slot. Computes sha256 + chunk count up front."""
        path = Path(source_path)
        if not path.exists():
            raise FileNotFoundError(f"source not found: {source_path}")
        total_size = path.stat().st_size
        sha = self._sha256_file(path)
        num_chunks = max(1, (total_size + chunk_size - 1) // chunk_size)
        now = time.time()
        rec = DownloadRecord(
            download_id=secrets.token_urlsafe(16),
            user_id=user_id,
            group_id=group_id,
            version=version,
            format=format,
            source_path=str(path),
            total_size=total_size,
            sha256=sha,
            chunk_size=chunk_size,
            num_chunks=num_chunks,
            created_at=now,
            expires_at=now + self.presign_ttl,
        )
        with self._lock:
            self._records[rec.download_id] = rec
        return rec

    @staticmethod
    def _sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while True:
                buf = f.read(chunk)
                if not buf:
                    break
                h.update(buf)
        return h.hexdigest()

    def sign_chunk_url(self, download_id: str, chunk_index: int) -> tuple[str, float]:
        rec = self._records.get(download_id)
        if rec is None:
            raise KeyError(download_id)
        expires = time.time() + self.presign_ttl
        sig = self._sign(download_id, chunk_index, expires)
        return (
            f"/api/downloads/{download_id}/chunk/{chunk_index}"
            f"?expires={int(expires)}&sig={sig}",
            expires,
        )

    def verify(self, download_id: str, chunk_index: int, expires: float, sig: str) -> bool:
        if expires < time.time():
            return False
        expected = self._sign(download_id, chunk_index, expires)
        return hmac.compare_digest(expected, sig)

    def _sign(self, download_id: str, chunk_index: int, expires: float) -> str:
        msg = f"{download_id}:{chunk_index}:{int(expires)}".encode()
        return hmac.new(self.secret_key, msg, hashlib.sha256).hexdigest()

    def get(self, download_id: str) -> DownloadRecord | None:
        return self._records.get(download_id)

    def record_chunk_served(self, download_id: str, n: int) -> None:
        rec = self._records.get(download_id)
        if rec is None:
            return
        rec.bytes_served += n

    def mark_complete(self, download_id: str) -> None:
        rec = self._records.get(download_id)
        if rec is None:
            return
        rec.status = "completed"
        rec.completed_at = time.time()

    def abort(self, download_id: str) -> None:
        rec = self._records.get(download_id)
        if rec is not None:
            rec.status = "aborted"
        with self._lock:
            self._records.pop(download_id, None)


_download_manager: DownloadManager | None = None


def init_download_manager(
    config: dict | None = None,
    secret_key: bytes | None = None,
) -> DownloadManager:
    """Build the singleton download manager from config + secret key."""
    global _download_manager
    cfg = config or DEFAULT_CONFIG.get("uploads", {})
    key = secret_key or os.environ.get("SECRET_KEY", "astra-dev-secret").encode()
    _download_manager = DownloadManager(
        secret_key=key,
        presign_ttl=cfg.get("presign_ttl_seconds", 3600),
    )
    return _download_manager


def get_download_manager() -> DownloadManager:
    if _download_manager is None:
        raise RuntimeError(
            "Download manager not initialized. The application lifespan must "
            "call init_download_manager() during startup."
        )
    return _download_manager
