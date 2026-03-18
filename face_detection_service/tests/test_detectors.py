"""Unit tests for individual face detectors."""

from __future__ import annotations

import numpy as np
import pytest

from face_detection_service.detectors.base import FaceDetection


# ── Haar Cascade ─────────────────────────────────────────────────────────────

class TestHaarCascadeDetector:
    """Tests for HaarCascadeDetector."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        from face_detection_service.detectors.haar import HaarCascadeDetector
        self.detector = HaarCascadeDetector()

    def test_name(self):
        assert self.detector.name == "haar"

    def test_detect_returns_list(self, face_image: np.ndarray):
        result = self.detector.detect(face_image)
        assert isinstance(result, list)
        for det in result:
            assert isinstance(det, FaceDetection)

    def test_detect_blank_image(self, blank_image: np.ndarray):
        result = self.detector.detect(blank_image)
        assert isinstance(result, list)

    def test_detection_fields(self, face_image: np.ndarray):
        results = self.detector.detect(face_image)
        for det in results:
            assert isinstance(det.bbox, tuple)
            assert len(det.bbox) == 4
            x, y, w, h = det.bbox
            assert w > 0
            assert h > 0
            assert 0.0 <= det.confidence <= 1.0

    def test_detect_invalid_image(self):
        invalid = np.zeros((0, 0, 3), dtype=np.uint8)
        # Should not crash — may return empty or raise
        try:
            result = self.detector.detect(invalid)
            assert isinstance(result, list)
        except Exception:
            pass  # acceptable to raise on truly invalid input


# ── Dlib HOG ─────────────────────────────────────────────────────────────────

class TestDlibHOGDetector:
    """Tests for DlibHOGDetector."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        dlib = pytest.importorskip("dlib")  # noqa: F841
        from face_detection_service.detectors.dlib_hog import DlibHOGDetector
        self.detector = DlibHOGDetector()

    def test_name(self):
        assert self.detector.name == "dlib_hog"

    def test_detect_returns_list(self, face_image: np.ndarray):
        result = self.detector.detect(face_image)
        assert isinstance(result, list)
        for det in result:
            assert isinstance(det, FaceDetection)

    def test_detect_blank_image(self, blank_image: np.ndarray):
        result = self.detector.detect(blank_image)
        assert isinstance(result, list)
        assert len(result) == 0

    def test_confidence_range(self, face_image: np.ndarray):
        for det in self.detector.detect(face_image):
            assert 0.0 <= det.confidence <= 1.0


# ── MediaPipe ────────────────────────────────────────────────────────────────

class TestMediaPipeDetector:
    """Tests for MediaPipeDetector."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        pytest.importorskip("mediapipe")
        from face_detection_service.detectors.mediapipe_det import MediaPipeDetector
        self.detector = MediaPipeDetector()

    def test_name(self):
        assert self.detector.name == "mediapipe"

    def test_detect_returns_list(self, face_image: np.ndarray):
        result = self.detector.detect(face_image)
        assert isinstance(result, list)
        for det in result:
            assert isinstance(det, FaceDetection)

    def test_detect_blank_image(self, blank_image: np.ndarray):
        result = self.detector.detect(blank_image)
        assert isinstance(result, list)
        assert len(result) == 0

    def test_landmarks_present_if_detected(self, face_image: np.ndarray):
        """MediaPipe should return landmarks when a face is found."""
        results = self.detector.detect(face_image)
        for det in results:
            assert isinstance(det.landmarks, list)


# ── Registry ─────────────────────────────────────────────────────────────────

class TestDetectorRegistry:
    """Tests for DetectorRegistry."""

    def test_register_and_get(self):
        from face_detection_service.detectors.registry import DetectorRegistry
        from face_detection_service.detectors.haar import HaarCascadeDetector

        reg = DetectorRegistry()
        reg.register("haar", HaarCascadeDetector)
        det = reg.get("haar")
        assert det.name == "haar"

    def test_unknown_detector_raises(self):
        from face_detection_service.detectors.registry import DetectorRegistry

        reg = DetectorRegistry()
        with pytest.raises(KeyError, match="Unknown detector"):
            reg.get("nonexistent")

    def test_list_available(self):
        from face_detection_service.detectors.registry import DetectorRegistry
        from face_detection_service.detectors.haar import HaarCascadeDetector

        reg = DetectorRegistry()
        reg.register("haar", HaarCascadeDetector)
        assert "haar" in reg.list_available()

    def test_lazy_loading(self):
        """Detector should not be instantiated until get() is called."""
        from face_detection_service.detectors.registry import DetectorRegistry
        from face_detection_service.detectors.haar import HaarCascadeDetector

        reg = DetectorRegistry()
        reg.register("haar", HaarCascadeDetector)
        assert "haar" not in reg._instances
        reg.get("haar")
        assert "haar" in reg._instances
