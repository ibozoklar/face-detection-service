"""Pydantic v2 request / response schemas for the face detection API."""

from pydantic import BaseModel, Field


# ── Request ──────────────────────────────────────────────────────────────────

class DetectionRequest(BaseModel):
    """JSON body for base64 image detection."""

    image_base64: str | None = Field(
        default=None,
        description="Base64-encoded image data (JPEG, PNG, BMP, or WebP).",
    )
    detector: str = Field(
        default="haar",
        description="Detector to use: 'haar', 'dlib_hog', or 'mediapipe'.",
    )


# ── Response components ──────────────────────────────────────────────────────

class FaceDetectionSchema(BaseModel):
    """Single detected face."""

    bbox: tuple[int, int, int, int] = Field(..., description="Bounding box as (x, y, width, height).")
    confidence: float = Field(..., ge=0.0, le=1.0)
    landmarks: list[tuple[int, int]] = Field(default_factory=list, description="Landmark (x, y) coordinates.")
    metadata: dict[str, object] = Field(default_factory=dict)


class ImageMetadata(BaseModel):
    """Basic information about the input image."""

    width: int
    height: int
    channels: int
    format: str | None = None


# ── Responses ────────────────────────────────────────────────────────────────

class DetectionResponse(BaseModel):
    """Successful detection response."""

    detections: list[FaceDetectionSchema]
    num_faces: int
    detector_used: str
    processing_time_ms: float = Field(..., description="Inference time in milliseconds.")
    image_metadata: ImageMetadata
    annotated_image_base64: str | None = Field(
        default=None,
        description="Base64-encoded annotated image (only when ?annotated=true).",
    )


class HealthResponse(BaseModel):
    """Health-check response."""

    status: str = "ok"
    available_detectors: list[str] = Field(default_factory=list)


class DetectorInfo(BaseModel):
    """Metadata about a single detector."""

    name: str
    description: str


class DetectorListResponse(BaseModel):
    """GET /detectors response."""

    detectors: list[DetectorInfo]


class ErrorResponse(BaseModel):
    """Standardised error payload."""

    detail: str
    request_id: str | None = None
