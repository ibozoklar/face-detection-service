"""Shared test fixtures."""

from __future__ import annotations

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

from face_detection_service.main import app


@pytest.fixture()
def face_image() -> np.ndarray:
    """Create a simple synthetic 300x300 BGR image with a bright rectangle
    in the centre that Haar cascade can pick up as a face-like region.

    Note: real detector accuracy tests should use actual face photos.
    This is a minimal fixture for smoke / integration testing.
    """
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    # Skin-tone-ish rectangle in centre
    img[80:220, 100:200] = (180, 200, 230)  # BGR
    # Dark regions for "eyes"
    img[120:140, 120:145] = (40, 30, 30)
    img[120:140, 155:180] = (40, 30, 30)
    return img


@pytest.fixture()
def blank_image() -> np.ndarray:
    """A plain black image with no face-like features."""
    return np.zeros((200, 200, 3), dtype=np.uint8)


@pytest.fixture()
async def async_client():
    """httpx AsyncClient wired to the FastAPI app."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
