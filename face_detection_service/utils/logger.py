"""Structured JSON logging with file rotation and SQLite detection storage."""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from typing import Any

from face_detection_service import config

# ── JSON file logger setup ───────────────────────────────────────────────────

_logger_initialised = False


def setup_logger() -> logging.Logger:
    """Configure the application-wide JSON logger (idempotent).

    Returns:
        Root application logger.
    """
    global _logger_initialised  # noqa: PLW0603
    logger = logging.getLogger("face_detection_service")

    if _logger_initialised:
        return logger

    logger.setLevel(config.LOG_LEVEL)

    # JSON file handler
    log_file = config.LOGS_DIR / "service.log"
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=config.LOG_FILE_MAX_BYTES,
        backupCount=config.LOG_FILE_BACKUP_COUNT,
    )
    file_handler.setFormatter(_JsonFormatter())
    logger.addHandler(file_handler)

    # Console handler (plain)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)s %(name)s — %(message)s")
    )
    logger.addHandler(console_handler)

    _logger_initialised = True
    return logger


class _JsonFormatter(logging.Formatter):
    """Format log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[1]:
            log_obj["exception"] = self.formatException(record.exc_info)
        # Merge any extra fields attached to the record
        for key in ("request_id", "user_id", "detector", "processing_time_ms", "num_faces"):
            value = getattr(record, key, None)
            if value is not None:
                log_obj[key] = value
        return json.dumps(log_obj, default=str)


# ── SQLite detection log store ───────────────────────────────────────────────

class DetectionLogStore:
    """Thread-safe SQLite store for detection request history.

    Uses WAL mode for concurrent read/write access.
    """

    _CREATE_TABLE = """
        CREATE TABLE IF NOT EXISTS detection_logs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            request_id  TEXT,
            user_id     TEXT,
            timestamp   TEXT    NOT NULL,
            detector    TEXT    NOT NULL,
            image_width INTEGER,
            image_height INTEGER,
            num_faces   INTEGER NOT NULL,
            processing_time_ms REAL NOT NULL
        )
    """

    def __init__(self, db_path: str | None = None) -> None:
        self._db_path = db_path or str(config.SQLITE_DB_PATH)
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(self._CREATE_TABLE)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        return conn

    def log_detection(
        self,
        *,
        request_id: str | None,
        user_id: str | None,
        detector: str,
        image_width: int | None = None,
        image_height: int | None = None,
        num_faces: int,
        processing_time_ms: float,
    ) -> None:
        """Insert a detection log record.

        Args:
            request_id: Unique request identifier.
            user_id: User identifier.
            detector: Name of the detector used.
            image_width: Input image width.
            image_height: Input image height.
            num_faces: Number of faces detected.
            processing_time_ms: Processing time in milliseconds.
        """
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO detection_logs
                    (request_id, user_id, timestamp, detector,
                     image_width, image_height, num_faces, processing_time_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (request_id, user_id, now, detector,
                 image_width, image_height, num_faces, processing_time_ms),
            )

    def get_logs(
        self,
        limit: int = 100,
        detector: str | None = None,
        user_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Query detection logs.

        Args:
            limit: Maximum number of records to return.
            detector: Filter by detector name.
            user_id: Filter by user ID.

        Returns:
            List of log records as dicts.
        """
        query = "SELECT * FROM detection_logs WHERE 1=1"
        params: list[Any] = []

        if detector:
            query += " AND detector = ?"
            params.append(detector)
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)

        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]
