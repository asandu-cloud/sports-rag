#!/usr/bin/env python3
"""
Shared Chroma backend helpers for local + deploy environments.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import chromadb


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHROMA_DIR = str(ROOT / "Index" / "chroma")


def env_first(*keys: str, default: Optional[str] = None) -> Optional[str]:
    for k in keys:
        v = os.environ.get(k)
        if v is not None and str(v).strip() != "":
            return str(v)
    return default


def env_bool(*keys: str, default: bool = False) -> bool:
    v = env_first(*keys)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


def _http_headers_from_env() -> dict:
    headers = {}
    token = env_first("CHROMA_AUTH_TOKEN", "CHROMA_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    raw = env_first("CHROMA_HTTP_HEADERS")
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                headers.update({str(k): str(v) for k, v in parsed.items()})
        except Exception:
            pass
    return headers


def get_chroma_client(chroma_dir: str = DEFAULT_CHROMA_DIR):
    """
    Local mode (default): chromadb.PersistentClient(path=...)
    Remote mode: set CHROMA_MODE=http and CHROMA_HTTP_HOST / CHROMA_HTTP_PORT
    """
    mode = (env_first("CHROMA_MODE", "CHROMA_CLIENT", default="local") or "local").lower()
    host = env_first("CHROMA_HTTP_HOST", "CHROMA_HOST")
    remote = mode in {"http", "remote"} or bool(host)

    if remote:
        port_raw = env_first("CHROMA_HTTP_PORT", "CHROMA_PORT", default="8000")
        try:
            port = int(str(port_raw))
        except Exception:
            port = 8000
        ssl = env_bool("CHROMA_HTTP_SSL", "CHROMA_SSL", default=False)
        kwargs = {
            "host": host or "localhost",
            "port": port,
            "ssl": ssl,
        }
        headers = _http_headers_from_env()
        if headers:
            kwargs["headers"] = headers
        tenant = env_first("CHROMA_TENANT")
        database = env_first("CHROMA_DATABASE")
        if tenant:
            kwargs["tenant"] = tenant
        if database:
            kwargs["database"] = database
        return chromadb.HttpClient(**kwargs)

    os.makedirs(chroma_dir, exist_ok=True)
    return chromadb.PersistentClient(path=chroma_dir)


def backend_description(chroma_dir: str = DEFAULT_CHROMA_DIR) -> str:
    mode = (env_first("CHROMA_MODE", "CHROMA_CLIENT", default="local") or "local").lower()
    host = env_first("CHROMA_HTTP_HOST", "CHROMA_HOST")
    remote = mode in {"http", "remote"} or bool(host)
    if remote:
        port = env_first("CHROMA_HTTP_PORT", "CHROMA_PORT", default="8000")
        ssl = env_bool("CHROMA_HTTP_SSL", "CHROMA_SSL", default=False)
        proto = "https" if ssl else "http"
        return f"{proto}://{host or 'localhost'}:{port}"
    return f"local:{chroma_dir}"
