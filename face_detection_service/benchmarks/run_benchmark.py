"""Benchmark script: compare detectors across different image sizes.

Usage:
    python -m face_detection_service.benchmarks.run_benchmark
"""

from __future__ import annotations

import logging
import tempfile
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from tabulate import tabulate

from face_detection_service.detectors.registry import create_default_registry

logger = logging.getLogger(__name__)

# ── Image sizes ──────────────────────────────────────────────────────────────

_SIZES: dict[str, tuple[int, int]] = {
    "small":  (320, 240),
    "medium": (640, 480),
    "large":  (1280, 960),
}

_FACE_IMAGE_URL = "https://thispersondoesnotexist.com"


# ── Image helpers ────────────────────────────────────────────────────────────

def _download_face_image() -> np.ndarray | None:
    """Download a real face photo from thispersondoesnotexist.com.

    Returns:
        BGR NumPy array on success, None on failure.
    """
    try:
        req = urllib.request.Request(
            _FACE_IMAGE_URL,
            headers={"User-Agent": "FaceDetectionBenchmark/1.0"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = resp.read()
        arr = np.frombuffer(data, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is not None:
            print(f"  Downloaded real face image ({image.shape[1]}x{image.shape[0]})")
        return image
    except Exception as exc:
        print(f"  Could not download face image: {exc}")
        return None


def _make_synthetic_image(width: int, height: int) -> np.ndarray:
    """Generate a synthetic image with a face-like bright region (fallback)."""
    img = np.random.randint(60, 120, (height, width, 3), dtype=np.uint8)

    cx, cy = width // 2, height // 2
    fw, fh = width // 6, height // 4
    img[cy - fh:cy + fh, cx - fw:cx + fw] = (180, 200, 230)

    ey = cy - fh // 3
    img[ey:ey + fh // 5, cx - fw // 2:cx - fw // 6] = (30, 30, 40)
    img[ey:ey + fh // 5, cx + fw // 6:cx + fw // 2] = (30, 30, 40)

    return img


def _prepare_images(source: np.ndarray | None) -> dict[str, np.ndarray]:
    """Resize source image to each benchmark size, or fall back to synthetic.

    Args:
        source: Downloaded real face image, or None.

    Returns:
        Dict of size_name → BGR image.
    """
    images: dict[str, np.ndarray] = {}
    for size_name, (w, h) in _SIZES.items():
        if source is not None:
            images[size_name] = cv2.resize(source, (w, h), interpolation=cv2.INTER_AREA)
        else:
            images[size_name] = _make_synthetic_image(w, h)
    return images


# ── Benchmark runner ─────────────────────────────────────────────────────────

def run(iterations: int = 5) -> list[dict[str, object]]:
    """Run benchmarks and return results.

    Args:
        iterations: Number of runs per detector/size combo (median is reported).

    Returns:
        List of result dicts for tabulation.
    """
    registry = create_default_registry()
    available = registry.list_available()

    if not available:
        print("No detectors available — nothing to benchmark.")
        return []

    # Download a real face image; fall back to synthetic on failure
    print("Preparing benchmark images …")
    source = _download_face_image()
    images = _prepare_images(source)
    using_real = source is not None
    if not using_real:
        print("  Using synthetic images (fallback).\n")
    else:
        print()

    results: list[dict[str, object]] = []

    for size_name, (w, h) in _SIZES.items():
        image = images[size_name]

        for det_name in available:
            detector = registry.get(det_name)
            times: list[float] = []
            face_counts: list[int] = []
            confidences: list[float] = []

            for _ in range(iterations):
                start = time.perf_counter()
                detections = detector.detect(image)
                elapsed = (time.perf_counter() - start) * 1000  # ms
                times.append(elapsed)
                face_counts.append(len(detections))
                confidences.extend(d.confidence for d in detections)

            sorted_times = sorted(times)
            median_time = sorted_times[len(sorted_times) // 2]
            min_time = sorted_times[0]
            max_time = sorted_times[-1]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

            results.append({
                "Detector": det_name,
                "Image": f"{size_name} ({w}x{h})",
                "Min ms": round(min_time, 2),
                "Median ms": round(median_time, 2),
                "Max ms": round(max_time, 2),
                "Faces": face_counts[0],
                "Avg conf": round(avg_confidence, 3),
            })

    return results


def main() -> None:
    """Entry point: run benchmarks and print table."""
    print("Running face detection benchmarks …\n")
    results = run()
    if results:
        print(tabulate(results, headers="keys", tablefmt="github"))
    print()


if __name__ == "__main__":
    main()
