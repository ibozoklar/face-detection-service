"""Base classes for face detection: abstract detector and detection result."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class FaceDetection:
    """Single face detection result.

    Attributes:
        x: Left edge of bounding box (pixels).
        y: Top edge of bounding box (pixels).
        width: Bounding box width (pixels).
        height: Bounding box height (pixels).
        confidence: Detection confidence score in [0, 1].
        landmarks: Optional dict of landmark name → (x, y) coordinates.
        metadata: Optional extra info from the detector.
    """

    x: int
    y: int
    width: int
    height: int
    confidence: float
    landmarks: dict[str, tuple[int, int]] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        """Return bounding box as (x, y, width, height)."""
        return (self.x, self.y, self.width, self.height)


class BaseFaceDetector(ABC):
    """Abstract base class that every face detector must implement.

    Subclasses must override ``name`` and ``detect``.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this detector (e.g. 'haar', 'dlib_hog')."""

    @abstractmethod
    def detect(self, image: np.ndarray) -> list[FaceDetection]:
        """Detect faces in a BGR NumPy image.

        Args:
            image: Input image as a NumPy array in BGR colour order.

        Returns:
            List of FaceDetection results (may be empty).
        """

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"
