"""Dlib HOG + SVM face detector."""

from __future__ import annotations

import cv2
import dlib
import numpy as np

from face_detection_service import config
from face_detection_service.detectors.base import BaseFaceDetector, FaceDetection


class DlibHOGDetector(BaseFaceDetector):
    """Face detection using dlib's HOG-based frontal face detector.

    Uses Histogram of Oriented Gradients features with a linear SVM classifier.
    Detection scores are normalised to [0, 1] for consistency with other detectors.
    """

    def __init__(self) -> None:
        self._detector = dlib.get_frontal_face_detector()
        self._upsample = config.DLIB_UPSAMPLE_NUM_TIMES

    @property
    def name(self) -> str:
        return "dlib_hog"

    def detect(self, image: np.ndarray) -> list[FaceDetection]:
        """Detect faces in a BGR image.

        Args:
            image: Input image as a NumPy array (BGR).

        Returns:
            List of FaceDetection results.
        """
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        rects, scores, _ = self._detector.run(rgb, self._upsample)

        detections: list[FaceDetection] = []
        for rect, score in zip(rects, scores):
            x = max(0, rect.left())
            y = max(0, rect.top())
            w = rect.right() - x
            h = rect.bottom() - y

            # Normalise raw SVM score to ~[0, 1] via a simple sigmoid
            confidence = 1.0 / (1.0 + _exp_safe(-score))

            detections.append(
                FaceDetection(
                    x=int(x),
                    y=int(y),
                    width=int(w),
                    height=int(h),
                    confidence=round(confidence, 4),
                )
            )
        return detections


def _exp_safe(x: float) -> float:
    """Clamped exp to avoid overflow."""
    import math
    return math.exp(max(-500.0, min(500.0, x)))
