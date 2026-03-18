"""Detector registry: maps string keys to detector instances with lazy loading."""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from face_detection_service.detectors.base import BaseFaceDetector

logger = logging.getLogger(__name__)


class DetectorRegistry:
    """Thread-safe registry that maps detector names to lazy-loaded instances.

    Usage:
        registry = DetectorRegistry()
        registry.register("haar", HaarCascadeDetector)
        detector = registry.get("haar")   # instantiated on first access
    """

    def __init__(self) -> None:
        self._classes: dict[str, type[BaseFaceDetector]] = {}
        self._instances: dict[str, BaseFaceDetector] = {}
        self._lock = threading.Lock()
        self._failed: dict[str, str] = {}  # name → error message

    # ── public API ───────────────────────────────────────────────────────

    def register(self, name: str, detector_class: type[BaseFaceDetector]) -> None:
        """Register a detector class under the given name."""
        self._classes[name] = detector_class
        logger.info("Registered detector class: %s", name)

    def get(self, name: str) -> BaseFaceDetector:
        """Return a detector instance, creating it lazily on first call.

        Args:
            name: Registered detector key.

        Returns:
            Detector instance.

        Raises:
            KeyError: If *name* was never registered.
            RuntimeError: If the detector previously failed to load.
        """
        if name in self._failed:
            raise RuntimeError(
                f"Detector '{name}' failed to load: {self._failed[name]}"
            )

        if name not in self._classes:
            available = ", ".join(self.list_available()) or "(none)"
            raise KeyError(
                f"Unknown detector '{name}'. Available: {available}"
            )

        if name not in self._instances:
            with self._lock:
                # double-check after acquiring the lock
                if name not in self._instances:
                    self._instances[name] = self._create(name)

        return self._instances[name]

    def list_available(self) -> list[str]:
        """Return names of detectors that are registered and not failed."""
        return [n for n in self._classes if n not in self._failed]

    def list_all(self) -> list[str]:
        """Return all registered detector names (including failed ones)."""
        return list(self._classes.keys())

    # ── internals ────────────────────────────────────────────────────────

    def _create(self, name: str) -> BaseFaceDetector:
        """Instantiate a detector, recording failures gracefully."""
        try:
            instance = self._classes[name]()
            logger.info("Loaded detector: %s", name)
            return instance
        except Exception as exc:
            self._failed[name] = str(exc)
            logger.error("Failed to load detector '%s': %s", name, exc)
            raise RuntimeError(
                f"Detector '{name}' failed to load: {exc}"
            ) from exc


def create_default_registry() -> DetectorRegistry:
    """Build a registry pre-populated with the three built-in detectors.

    Detectors that cannot be imported (missing dependency) are skipped
    with a warning rather than crashing the application.
    """
    registry = DetectorRegistry()

    # Haar Cascade — always available (ships with opencv)
    try:
        from face_detection_service.detectors.haar import HaarCascadeDetector
        registry.register("haar", HaarCascadeDetector)
    except ImportError as exc:
        logger.warning("Haar detector unavailable: %s", exc)

    # Dlib HOG
    try:
        from face_detection_service.detectors.dlib_hog import DlibHOGDetector
        registry.register("dlib_hog", DlibHOGDetector)
    except ImportError as exc:
        logger.warning("Dlib HOG detector unavailable: %s", exc)

    # MediaPipe
    try:
        from face_detection_service.detectors.mediapipe_det import MediaPipeDetector
        registry.register("mediapipe", MediaPipeDetector)
    except ImportError as exc:
        logger.warning("MediaPipe detector unavailable: %s", exc)

    return registry
