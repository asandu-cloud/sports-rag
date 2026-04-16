"""Raw provider payload archival.

Provides a minimal ``ObjectStorage`` protocol with two backends:

* ``LocalDiskStorage`` — writes payloads under a configurable root.
  Default for V1.
* ``S3Storage`` — stub for the S3-compatible future. Implemented just
  enough to prove the seam; flip to it by setting
  ``RAW_ARCHIVE_BACKEND=s3`` and wiring credentials.

``PayloadArchiver`` glues a backend to the Postgres metadata table
(``raw_payload_archive``) so callers get a durable pointer back to every
fetched response and can replay them later.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol, runtime_checkable

from sqlalchemy.orm import Session

from ..config import SETTINGS, Settings
from ..models.sync import RawPayloadArchive

logger = logging.getLogger(__name__)


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _params_digest(params: Mapping[str, Any]) -> str:
    canonical = json.dumps(params, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@runtime_checkable
class ObjectStorage(Protocol):
    """Minimal protocol shared by every backend."""

    backend_name: str

    def put(self, key: str, data: bytes, *, content_type: str = "application/octet-stream") -> str:
        """Persist ``data`` at ``key``. Returns a URI usable by ``get``."""

    def get(self, key: str) -> bytes:
        """Retrieve a previously-stored blob."""

    def exists(self, key: str) -> bool:
        """Return True if ``key`` is already stored."""


class LocalDiskStorage:
    """Filesystem-backed storage. Safe for tests + single-host deploys."""

    backend_name = "local"

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        # Normalise key, forbid traversal
        safe = key.lstrip("/")
        if ".." in Path(safe).parts:
            raise ValueError(f"Unsafe key: {key!r}")
        return self.root / safe

    def put(self, key: str, data: bytes, *, content_type: str = "application/octet-stream") -> str:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path.resolve().as_uri()

    def get(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def exists(self, key: str) -> bool:
        return self._path(key).exists()


class S3Storage:
    """S3 / S3-compatible backend. Thin stub for V1.

    ``boto3`` is not a hard dependency; import lazily so the rest of the
    platform runs without it. Uploads are gzip-transparent at the
    archiver layer (keys end with ``.json.gz``).
    """

    backend_name = "s3"

    def __init__(
        self,
        bucket: str,
        *,
        prefix: str = "",
        region: Optional[str] = None,
        endpoint_url: Optional[str] = None,
    ):
        if not bucket:
            raise ValueError("S3Storage requires a bucket name")
        try:
            import boto3  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover - exercised only when boto3 missing
            raise RuntimeError(
                "boto3 is required for S3Storage. Install it with `pip install boto3`."
            ) from exc
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.client = boto3.client("s3", region_name=region, endpoint_url=endpoint_url)

    def _key(self, key: str) -> str:
        safe = key.lstrip("/")
        return f"{self.prefix}/{safe}" if self.prefix else safe

    def put(self, key: str, data: bytes, *, content_type: str = "application/octet-stream") -> str:
        full_key = self._key(key)
        self.client.put_object(Bucket=self.bucket, Key=full_key, Body=data, ContentType=content_type)
        return f"s3://{self.bucket}/{full_key}"

    def get(self, key: str) -> bytes:
        full_key = self._key(key)
        resp = self.client.get_object(Bucket=self.bucket, Key=full_key)
        return resp["Body"].read()

    def exists(self, key: str) -> bool:
        full_key = self._key(key)
        try:
            self.client.head_object(Bucket=self.bucket, Key=full_key)
            return True
        except Exception:
            return False


def build_storage(settings: Optional[Settings] = None) -> ObjectStorage:
    """Factory: returns the backend matching current settings."""
    s = settings or SETTINGS
    if s.raw_archive_backend == "s3":
        if not s.s3_bucket:
            logger.warning("RAW_ARCHIVE_BACKEND=s3 but S3_BUCKET is unset; falling back to local disk.")
        else:
            return S3Storage(
                bucket=s.s3_bucket,
                prefix=s.s3_prefix or "",
                region=s.s3_region,
                endpoint_url=s.s3_endpoint_url,
            )
    return LocalDiskStorage(s.raw_archive_root)


@dataclass(frozen=True)
class ArchiveResult:
    """Return value of ``PayloadArchiver.archive_json``."""

    archive_id: int
    storage_uri: str
    payload_digest: str
    was_new: bool


class PayloadArchiver:
    """Writes raw payloads to object storage and records the pointer row."""

    def __init__(self, session: Session, storage: Optional[ObjectStorage] = None, *, settings: Optional[Settings] = None):
        self.session = session
        self.settings = settings or SETTINGS
        self.storage = storage or build_storage(self.settings)

    def archive_json(
        self,
        *,
        provider: str,
        endpoint: str,
        params: Mapping[str, Any],
        payload: Any,
        sync_run_id: Optional[int] = None,
        compress: bool = True,
    ) -> ArchiveResult:
        """Serialize ``payload`` as JSON, store it, and register metadata.

        Idempotent on ``(provider, endpoint, params_digest, payload_digest)``:
        if the exact same payload bytes were already archived, we return
        the existing row instead of writing a duplicate.
        """
        raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        # Hash on the canonicalised JSON bytes so the dedup digest stays
        # stable regardless of compression metadata (gzip writes a mtime).
        payload_digest = _sha256_hex(raw)
        content_type = "application/json"
        suffix = ".json"
        body = raw
        if compress:
            body = gzip.compress(raw, mtime=0)
            content_type = "application/gzip"
            suffix = ".json.gz"
        params_dig = _params_digest(params)

        # Short-circuit duplicates to save both storage and metadata rows
        existing = (
            self.session.query(RawPayloadArchive)
            .filter(
                RawPayloadArchive.provider == provider,
                RawPayloadArchive.endpoint == endpoint,
                RawPayloadArchive.params_digest == params_dig,
                RawPayloadArchive.payload_digest == payload_digest,
            )
            .order_by(RawPayloadArchive.id.desc())
            .first()
        )
        if existing is not None:
            return ArchiveResult(
                archive_id=existing.id,
                storage_uri=existing.storage_uri,
                payload_digest=payload_digest,
                was_new=False,
            )

        fetched_at = datetime.now(timezone.utc)
        ts_key = fetched_at.strftime("%Y/%m/%d/%H%M%S")
        key = f"{provider}/{endpoint.strip('/')}/{ts_key}_{params_dig[:12]}_{payload_digest[:12]}{suffix}"
        storage_uri = self.storage.put(key, body, content_type=content_type)

        row = RawPayloadArchive(
            provider=provider,
            endpoint=endpoint,
            params_digest=params_dig,
            params=dict(params),
            storage_backend=self.storage.backend_name,
            storage_uri=storage_uri,
            payload_digest=payload_digest,
            bytes_size=len(body),
            content_type=content_type,
            fetched_at=fetched_at,
            sync_run_id=sync_run_id,
        )
        self.session.add(row)
        self.session.flush()
        return ArchiveResult(
            archive_id=row.id,
            storage_uri=storage_uri,
            payload_digest=payload_digest,
            was_new=True,
        )

    def load_json(self, archive_id: int) -> Any:
        """Rehydrate a previously archived payload for replay/debugging."""
        row = self.session.get(RawPayloadArchive, archive_id)
        if row is None:
            raise KeyError(f"RawPayloadArchive id={archive_id} not found")
        uri = row.storage_uri
        if uri.startswith("file://"):
            key = Path(uri.replace("file://", "")).relative_to(self.settings.raw_archive_root).as_posix()
        elif uri.startswith("s3://"):
            # s3://bucket/key - strip scheme + bucket to derive key
            _, _, rest = uri.partition("://")
            _, _, key = rest.partition("/")
        else:
            key = uri
        data = self.storage.get(key)
        if row.content_type == "application/gzip":
            data = gzip.decompress(data)
        return json.loads(data.decode("utf-8"))
