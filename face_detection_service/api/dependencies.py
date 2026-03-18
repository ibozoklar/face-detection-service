"""FastAPI dependencies for request tracing and shared singletons."""

from __future__ import annotations

import uuid
from functools import lru_cache

from fastapi import Header

from face_detection_service.detectors.registry import DetectorRegistry, create_default_registry
from face_detection_service.utils.logger import DetectionLogStore


def get_request_id(x_request_id: str | None = Header(default=None)) -> str:
    """Return the request ID from header or generate a new UUID."""
    return x_request_id or uuid.uuid4().hex


def get_user_id(x_user_id: str | None = Header(default=None)) -> str:
    """Return the user ID from header or 'anonymous'."""
    return x_user_id or "anonymous"


@lru_cache(maxsize=1)
def get_detector_registry() -> DetectorRegistry:
    """Singleton detector registry (created once per worker process)."""
    return create_default_registry()


@lru_cache(maxsize=1)
def get_log_store() -> DetectionLogStore:
    """Singleton SQLite log store (created once per worker process)."""
    return DetectionLogStore()
