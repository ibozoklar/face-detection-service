"""Application configuration via environment variables with sensible defaults."""

import os
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR: Path = Path(__file__).resolve().parent
LOGS_DIR: Path = Path(os.getenv("LOGS_DIR", str(BASE_DIR / "logs")))
SQLITE_DB_PATH: Path = Path(os.getenv("SQLITE_DB_PATH", str(LOGS_DIR / "detections.db")))

# ── Server ───────────────────────────────────────────────────────────────────
HOST: str = os.getenv("HOST", "0.0.0.0")
PORT: int = int(os.getenv("PORT", "8000"))
WORKERS: int = int(os.getenv("WORKERS", "1"))

# ── Image constraints ────────────────────────────────────────────────────────
MAX_IMAGE_SIZE_MB: int = int(os.getenv("MAX_IMAGE_SIZE_MB", "10"))
MAX_IMAGE_DIMENSION: int = int(os.getenv("MAX_IMAGE_DIMENSION", "4096"))
ALLOWED_IMAGE_FORMATS: list[str] = os.getenv(
    "ALLOWED_IMAGE_FORMATS", "jpeg,jpg,png,bmp,webp"
).split(",")

# ── Haar Cascade ─────────────────────────────────────────────────────────────
HAAR_SCALE_FACTOR: float = float(os.getenv("HAAR_SCALE_FACTOR", "1.1"))
HAAR_MIN_NEIGHBORS: int = int(os.getenv("HAAR_MIN_NEIGHBORS", "5"))
HAAR_MIN_SIZE: tuple[int, int] = tuple(
    int(x) for x in os.getenv("HAAR_MIN_SIZE", "30,30").split(",")
)  # type: ignore[assignment]

# ── Dlib HOG ─────────────────────────────────────────────────────────────────
DLIB_UPSAMPLE_NUM_TIMES: int = int(os.getenv("DLIB_UPSAMPLE_NUM_TIMES", "1"))

# ── MediaPipe ────────────────────────────────────────────────────────────────
MEDIAPIPE_MIN_DETECTION_CONFIDENCE: float = float(
    os.getenv("MEDIAPIPE_MIN_DETECTION_CONFIDENCE", "0.5")
)
MEDIAPIPE_MODEL_SELECTION: int = int(os.getenv("MEDIAPIPE_MODEL_SELECTION", "0"))
# 0 = short-range (< 2m), 1 = full-range (< 5m)

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE_MAX_BYTES: int = int(os.getenv("LOG_FILE_MAX_BYTES", str(5 * 1024 * 1024)))  # 5 MB
LOG_FILE_BACKUP_COUNT: int = int(os.getenv("LOG_FILE_BACKUP_COUNT", "3"))

# ── Thread pool ──────────────────────────────────────────────────────────────
THREAD_POOL_MAX_WORKERS: int = int(os.getenv("THREAD_POOL_MAX_WORKERS", "4"))

# ── Ensure directories exist ────────────────────────────────────────────────
LOGS_DIR.mkdir(parents=True, exist_ok=True)
