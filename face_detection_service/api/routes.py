"""API routes: /detect, /health, /detectors."""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile

from face_detection_service import config
from face_detection_service.api.dependencies import (
    get_detector_registry,
    get_log_store,
    get_request_id,
    get_user_id,
)
from face_detection_service.detectors.registry import DetectorRegistry
from face_detection_service.models.schemas import (
    DetectionRequest,
    DetectionResponse,
    DetectorInfo,
    DetectorListResponse,
    FaceDetectionSchema,
    HealthResponse,
    ImageMetadata,
)
from face_detection_service.utils.image_utils import (
    annotate_image,
    decode_base64,
    decode_upload,
    get_image_metadata,
    image_to_base64,
    validate_image,
)
from face_detection_service.utils.logger import DetectionLogStore

logger = logging.getLogger("face_detection_service")
router = APIRouter()

_executor = ThreadPoolExecutor(max_workers=config.THREAD_POOL_MAX_WORKERS)


# ── POST /detect ─────────────────────────────────────────────────────────────

@router.post("/detect", response_model=DetectionResponse)
async def detect_faces(
    request: Request,
    detector: str = Query(default="haar", description="Detector: haar, dlib_hog, mediapipe"),
    annotated: bool = Query(default=False, description="Return annotated image"),
    request_id: Annotated[str, Depends(get_request_id)] = "",
    user_id: Annotated[str, Depends(get_user_id)] = "",
    registry: Annotated[DetectorRegistry, Depends(get_detector_registry)] = None,  # type: ignore[assignment]
    log_store: Annotated[DetectionLogStore, Depends(get_log_store)] = None,  # type: ignore[assignment]
) -> DetectionResponse:
    """Detect faces in an uploaded image file or a base64-encoded image.

    - **multipart/form-data**: send a `file` field with the image.
    - **application/json**: send `{"image_base64": "..."}` in the body.

    The detector is always selected via the `?detector=` query parameter.
    """
    content_type = request.headers.get("content-type", "")

    if "multipart/form-data" in content_type:
        # File upload path
        form = await request.form()
        file = form.get("file")
        if file is None:
            raise HTTPException(status_code=422, detail="Missing 'file' field in multipart form data.")
        try:
            image = await decode_upload(file)  # type: ignore[arg-type]
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    else:
        # JSON / base64 path
        try:
            body = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Invalid JSON body: {exc}") from exc

        parsed = DetectionRequest(**body)
        if not parsed.image_base64:
            raise HTTPException(status_code=422, detail="Missing 'image_base64' field in JSON body.")
        try:
            image = decode_base64(parsed.image_base64)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return await _run_detection(image, detector, annotated, request_id, user_id, registry, log_store)


async def _run_detection(
    image,
    detector_name: str,
    annotated: bool,
    request_id: str,
    user_id: str,
    registry: DetectorRegistry,
    log_store: DetectionLogStore,
) -> DetectionResponse:
    """Shared detection logic for both upload and base64 endpoints."""
    # Validate
    try:
        validate_image(image)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # Get detector
    try:
        det = registry.get(detector_name)
    except (KeyError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Run detection in thread pool (CPU-bound)
    loop = asyncio.get_running_loop()
    start = time.perf_counter()
    detections = await loop.run_in_executor(_executor, det.detect, image)
    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)

    # Image metadata
    meta = get_image_metadata(image)

    # Optional annotation
    annotated_b64: str | None = None
    if annotated:
        ann_img = annotate_image(image, detections)
        annotated_b64 = image_to_base64(ann_img)

    # Log to SQLite (fire-and-forget in executor)
    _num_faces = len(detections)
    loop.run_in_executor(
        _executor,
        lambda: log_store.log_detection(
            request_id=request_id,
            user_id=user_id,
            detector=detector_name,
            image_width=meta["width"],
            image_height=meta["height"],
            num_faces=_num_faces,
            processing_time_ms=elapsed_ms,
        ),
    )

    # Structured log
    logger.info(
        "Detection complete",
        extra={
            "request_id": request_id,
            "user_id": user_id,
            "detector": detector_name,
            "num_faces": len(detections),
            "processing_time_ms": elapsed_ms,
        },
    )

    return DetectionResponse(
        detections=[
            FaceDetectionSchema(
                bbox=d.bbox,
                confidence=d.confidence,
                landmarks=[list(lm) for lm in d.landmarks] if d.landmarks else None,
                metadata=d.metadata,
            )
            for d in detections
        ],
        num_faces=len(detections),
        detector_used=detector_name,
        processing_time_ms=elapsed_ms,
        image_metadata=ImageMetadata(
            width=meta["width"],
            height=meta["height"],
            channels=meta["channels"],
        ),
        annotated_image_base64=annotated_b64,
    )


# ── GET /health ──────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse)
async def health_check(
    registry: Annotated[DetectorRegistry, Depends(get_detector_registry)] = None,  # type: ignore[assignment]
) -> HealthResponse:
    """Return service health and available detectors."""
    return HealthResponse(
        status="ok",
        available_detectors=registry.list_available(),
    )


# ── GET /detectors ───────────────────────────────────────────────────────────

_DETECTOR_DESCRIPTIONS: dict[str, str] = {
    "haar": "OpenCV Haar Cascade — fast classical Viola-Jones detector",
    "dlib_hog": "Dlib HOG + SVM — mid-tier accuracy with gradient features",
    "mediapipe": "MediaPipe BlazeFace — modern neural network with landmarks",
}


@router.get("/detectors", response_model=DetectorListResponse)
async def list_detectors(
    registry: Annotated[DetectorRegistry, Depends(get_detector_registry)] = None,  # type: ignore[assignment]
) -> DetectorListResponse:
    """List all available detectors with descriptions."""
    return DetectorListResponse(
        detectors=[
            DetectorInfo(
                name=name,
                description=_DETECTOR_DESCRIPTIONS.get(name, ""),
            )
            for name in registry.list_available()
        ]
    )
