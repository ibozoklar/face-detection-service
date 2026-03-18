"""MediaPipe (BlazeFace) face detector with landmark support."""

from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np

from face_detection_service import config
from face_detection_service.detectors.base import BaseFaceDetector, FaceDetection

# MediaPipe keypoint indices → human-readable names
_KEYPOINT_NAMES: dict[int, str] = {
    0: "right_eye",
    1: "left_eye",
    2: "nose_tip",
    3: "mouth_center",
    4: "right_ear_tragion",
    5: "left_ear_tragion",
}


class MediaPipeDetector(BaseFaceDetector):
    """Google MediaPipe BlazeFace detector.

    Provides 6 facial keypoints alongside bounding boxes.
    Lightweight neural network with native Apple Silicon support.
    """

    def __init__(self) -> None:
        self._min_confidence = config.MEDIAPIPE_MIN_DETECTION_CONFIDENCE
        self._model_selection = config.MEDIAPIPE_MODEL_SELECTION
        self._face_detection = mp.solutions.face_detection.FaceDetection(
            min_detection_confidence=self._min_confidence,
            model_selection=self._model_selection,
        )

    @property
    def name(self) -> str:
        return "mediapipe"

    def detect(self, image: np.ndarray) -> list[FaceDetection]:
        """Detect faces in a BGR image.

        Args:
            image: Input image as a NumPy array (BGR).

        Returns:
            List of FaceDetection results with 6-point landmarks.
        """
        h, w, _ = image.shape
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = self._face_detection.process(rgb)

        if not results.detections:
            return []

        detections: list[FaceDetection] = []
        for det in results.detections:
            bbox = det.location_data.relative_bounding_box

            # Convert normalised coords to pixel values
            x = max(0, int(bbox.xmin * w))
            y = max(0, int(bbox.ymin * h))
            bw = int(bbox.width * w)
            bh = int(bbox.height * h)

            # Clamp to image bounds
            bw = min(bw, w - x)
            bh = min(bh, h - y)

            # Extract 6 keypoints
            landmarks: dict[str, tuple[int, int]] = {}
            for idx, kp in enumerate(det.location_data.relative_keypoints):
                kp_name = _KEYPOINT_NAMES.get(idx, f"keypoint_{idx}")
                landmarks[kp_name] = (int(kp.x * w), int(kp.y * h))

            detections.append(
                FaceDetection(
                    x=x,
                    y=y,
                    width=bw,
                    height=bh,
                    confidence=round(det.score[0], 4),
                    landmarks=landmarks,
                )
            )
        return detections

    def __del__(self) -> None:
        """Release MediaPipe resources."""
        if hasattr(self, "_face_detection"):
            self._face_detection.close()
