"""API endpoint integration tests."""

from __future__ import annotations

import base64
from io import BytesIO

import cv2
import numpy as np
import pytest

pytestmark = pytest.mark.asyncio


def _encode_image_bytes(image: np.ndarray, fmt: str = ".jpg") -> bytes:
    """Encode a NumPy image to raw bytes."""
    _, buf = cv2.imencode(fmt, image)
    return buf.tobytes()


def _encode_image_b64(image: np.ndarray, fmt: str = ".jpg") -> str:
    """Encode a NumPy image to a base64 string."""
    return base64.b64encode(_encode_image_bytes(image, fmt)).decode()


# ── POST /detect (file upload) ───────────────────────────────────────────────

class TestDetectUpload:
    async def test_upload_jpeg(self, async_client, face_image):
        raw = _encode_image_bytes(face_image)
        resp = await async_client.post(
            "/detect",
            files={"file": ("face.jpg", BytesIO(raw), "image/jpeg")},
            params={"detector": "haar"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "detections" in data
        assert data["detector_used"] == "haar"
        assert data["processing_time_ms"] >= 0
        assert data["image_metadata"]["width"] == 300
        assert data["image_metadata"]["height"] == 300

    async def test_upload_invalid_detector(self, async_client, face_image):
        raw = _encode_image_bytes(face_image)
        resp = await async_client.post(
            "/detect",
            files={"file": ("face.jpg", BytesIO(raw), "image/jpeg")},
            params={"detector": "nonexistent"},
        )
        assert resp.status_code == 400

    async def test_upload_corrupt_file(self, async_client):
        resp = await async_client.post(
            "/detect",
            files={"file": ("bad.jpg", BytesIO(b"not-an-image"), "image/jpeg")},
        )
        assert resp.status_code == 422

    async def test_upload_with_annotation(self, async_client, face_image):
        raw = _encode_image_bytes(face_image)
        resp = await async_client.post(
            "/detect",
            files={"file": ("face.jpg", BytesIO(raw), "image/jpeg")},
            params={"detector": "haar", "annotated": "true"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["annotated_image_base64"] is not None

    async def test_request_id_header(self, async_client, face_image):
        raw = _encode_image_bytes(face_image)
        resp = await async_client.post(
            "/detect",
            files={"file": ("face.jpg", BytesIO(raw), "image/jpeg")},
            headers={"X-Request-ID": "test-123", "X-User-ID": "user-abc"},
        )
        assert resp.status_code == 200


# ── POST /detect (base64 JSON) ───────────────────────────────────────────────

class TestDetectBase64:
    async def test_base64_detect(self, async_client, face_image):
        b64 = _encode_image_b64(face_image)
        resp = await async_client.post(
            "/detect",
            json={"image_base64": b64},
            params={"detector": "haar"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["detector_used"] == "haar"
        assert isinstance(data["detections"], list)

    async def test_base64_invalid_data(self, async_client):
        resp = await async_client.post(
            "/detect",
            json={"image_base64": "not-valid-base64!!!", "detector": "haar"},
        )
        assert resp.status_code == 422


# ── GET /health ──────────────────────────────────────────────────────────────

class TestHealth:
    async def test_health(self, async_client):
        resp = await async_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert isinstance(data["available_detectors"], list)
        assert len(data["available_detectors"]) > 0


# ── GET /detectors ───────────────────────────────────────────────────────────

class TestDetectors:
    async def test_list_detectors(self, async_client):
        resp = await async_client.get("/detectors")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data["detectors"], list)
        names = [d["name"] for d in data["detectors"]]
        assert "haar" in names
