"""Haar Cascade face detector using OpenCV's built-in classifier."""

from __future__ import annotations

import cv2
import numpy as np

from face_detection_service import config
from face_detection_service.detectors.base import BaseFaceDetector, FaceDetection


class HaarCascadeDetector(BaseFaceDetector):
    """Viola-Jones face detection via OpenCV Haar cascades.

    The pre-trained XML classifier ships with OpenCV — no extra downloads needed.
    """

    def __init__(self) -> None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"  # type: ignore[attr-defined]
        self._classifier = cv2.CascadeClassifier(cascade_path)
        if self._classifier.empty():
            raise RuntimeError(f"Failed to load Haar cascade from {cascade_path}")

        self._scale_factor = config.HAAR_SCALE_FACTOR
        self._min_neighbors = config.HAAR_MIN_NEIGHBORS
        self._min_size = config.HAAR_MIN_SIZE

    @property
    def name(self) -> str:
        return "haar"

    def detect(self, image: np.ndarray) -> list[FaceDetection]:
        """Detect faces in a BGR image.

        Args:
            image: Input image as a NumPy array (BGR).

        Returns:
            List of FaceDetection results.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        try:
            rects, reject_levels, weights = self._classifier.detectMultiScale3(
                gray,
                scaleFactor=self._scale_factor,
                minNeighbors=self._min_neighbors,
                minSize=self._min_size,
                flags=cv2.CASCADE_SCALE_IMAGE,
                outputRejectLevels=True,
            )
        except cv2.error:
            # Fallback to detectMultiScale if detectMultiScale3 fails
            rects = self._classifier.detectMultiScale(
                gray,
                scaleFactor=self._scale_factor,
                minNeighbors=self._min_neighbors,
                minSize=self._min_size,
                flags=cv2.CASCADE_SCALE_IMAGE,
            )
            weights = None

        detections: list[FaceDetection] = []
        if len(rects) == 0:
            return detections

        # Normalize weights to [0, 1] via min-max scaling
        has_weights = weights is not None and len(weights) > 0
        if has_weights:
            w_array = np.array(weights, dtype=np.float64).flatten()
            w_min, w_max = float(w_array.min()), float(w_array.max())
            if w_max > w_min:
                norm_weights = (w_array - w_min) / (w_max - w_min)
                # Clamp to [0.5, 1.0] so even the weakest detection has reasonable confidence
                norm_weights = norm_weights * 0.5 + 0.5
            else:
                # All weights identical — use sigmoid fallback
                norm_weights = 1.0 / (1.0 + np.exp(-w_array))

        for i, (x, y, w, h) in enumerate(rects):
            if has_weights:
                confidence = round(float(norm_weights[i]), 4)
            else:
                confidence = 1.0  # fallback when detectMultiScale3 unavailable

            detections.append(
                FaceDetection(
                    bbox=(int(x), int(y), int(w), int(h)),
                    confidence=confidence,
                )
            )
        return detections
