"""Image decoding, validation, annotation, and conversion utilities."""

from __future__ import annotations

import base64
from io import BytesIO
from typing import TYPE_CHECKING

import cv2
import numpy as np

from face_detection_service import config

if TYPE_CHECKING:
    from fastapi import UploadFile

    from face_detection_service.detectors.base import FaceDetection


# ── Decode ───────────────────────────────────────────────────────────────────

async def decode_upload(file: UploadFile) -> np.ndarray:
    """Decode a FastAPI UploadFile to a BGR NumPy array.

    Args:
        file: Uploaded image file.

    Returns:
        Image as a BGR NumPy array.

    Raises:
        ValueError: If the file cannot be decoded as an image.
    """
    raw = await file.read()
    return _bytes_to_image(raw)


def decode_base64(b64_string: str) -> np.ndarray:
    """Decode a base64-encoded image string to a BGR NumPy array.

    Args:
        b64_string: Base64 image data (with or without data-URI prefix).

    Returns:
        Image as a BGR NumPy array.

    Raises:
        ValueError: If decoding fails.
    """
    # Strip optional data-URI prefix (e.g. "data:image/png;base64,")
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]

    try:
        raw = base64.b64decode(b64_string)
    except Exception as exc:
        raise ValueError(f"Invalid base64 data: {exc}") from exc

    return _bytes_to_image(raw)


def _bytes_to_image(raw: bytes) -> np.ndarray:
    """Convert raw bytes to a BGR NumPy image."""
    arr = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image — unsupported or corrupt file.")
    return image


# ── Validate ─────────────────────────────────────────────────────────────────

def validate_image(image: np.ndarray) -> None:
    """Validate image dimensions and size.

    Args:
        image: BGR NumPy array.

    Raises:
        ValueError: If the image exceeds configured limits.
    """
    h, w = image.shape[:2]
    max_dim = config.MAX_IMAGE_DIMENSION
    if h > max_dim or w > max_dim:
        raise ValueError(
            f"Image dimensions {w}x{h} exceed maximum {max_dim}x{max_dim}."
        )

    size_mb = image.nbytes / (1024 * 1024)
    if size_mb > config.MAX_IMAGE_SIZE_MB:
        raise ValueError(
            f"Image size {size_mb:.1f} MB exceeds maximum {config.MAX_IMAGE_SIZE_MB} MB."
        )


# ── Annotate ─────────────────────────────────────────────────────────────────

def annotate_image(
    image: np.ndarray,
    detections: list[FaceDetection],
) -> np.ndarray:
    """Draw bounding boxes and landmarks on a copy of the image.

    Args:
        image: Original BGR image.
        detections: List of FaceDetection results.

    Returns:
        Annotated image copy.
    """
    annotated = image.copy()
    for det in detections:
        x, y, w, h = det.bbox
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)

        label = f"{det.confidence:.2f}"
        cv2.putText(
            annotated, label, (x, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
        )

        for (lx, ly) in det.landmarks:
            cv2.circle(annotated, (lx, ly), 3, (0, 0, 255), -1)

    return annotated


# ── Conversion ───────────────────────────────────────────────────────────────

def image_to_base64(image: np.ndarray, fmt: str = ".jpg") -> str:
    """Encode a BGR image to a base64 string.

    Args:
        image: BGR NumPy array.
        fmt: Image format extension (e.g. '.jpg', '.png').

    Returns:
        Base64-encoded string.
    """
    success, buf = cv2.imencode(fmt, image)
    if not success:
        raise RuntimeError("Failed to encode image.")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def get_image_metadata(image: np.ndarray) -> dict[str, object]:
    """Extract basic metadata from an image array.

    Args:
        image: BGR NumPy array.

    Returns:
        Dict with width, height, channels keys.
    """
    h, w = image.shape[:2]
    channels = image.shape[2] if image.ndim == 3 else 1
    return {
        "width": w,
        "height": h,
        "channels": channels,
    }
