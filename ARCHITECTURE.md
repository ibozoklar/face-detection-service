# Architectural Decisions

## 1. Overall Architecture

The service follows a **modular plugin architecture** built around two core design patterns:

**Strategy Pattern** — Each face detection algorithm implements a common `BaseFaceDetector` abstract class with a single `detect(image) → List[FaceDetection]` contract. This allows any detector to be swapped in without touching API logic, request handling, or logging. The caller never knows which concrete detector it's using — only that it receives a list of `FaceDetection` objects with bounding boxes, confidence scores, and optional landmarks.

**Registry Pattern** — A `DetectorRegistry` maps string keys (e.g., `"haar"`, `"dlib_hog"`, `"mediapipe"`) to detector classes. The registry handles instantiation, lazy loading, and graceful degradation. If a detector fails to load (e.g., missing system dependency), the registry logs the failure and continues operating with the remaining detectors. This means a broken dlib installation on one platform doesn't take down the entire service.

These two patterns together create a system where adding a new detector requires exactly two steps: write a class that extends `BaseFaceDetector`, and register it. No other file needs modification.

## 2. Detector Selection

Three detectors were chosen to represent a spectrum from classical computer vision to modern deep learning:

**Haar Cascade (OpenCV)** — The classical baseline. Uses Viola-Jones detection with pre-trained cascade classifiers. Chosen because it ships with OpenCV (zero additional dependencies), runs on CPU with minimal overhead, and provides a useful reference point for benchmarking. Limitations: high false positive rate, sensitive to rotation and occlusion.

**Dlib HOG** — A mid-tier detector using Histogram of Oriented Gradients features with a linear SVM classifier. Selected because it demonstrates feature engineering fundamentals (gradient computation, cell histograms, block normalization) rather than end-to-end learning. It strikes a balance between speed and accuracy. Dlib's CNN-based detector could be integrated as a future enhancement for higher accuracy at the cost of increased inference time.

**MediaPipe Face Detection** — Google's production-grade solution using a lightweight neural network (BlazeFace). Chosen for its speed (sub-millisecond on modern hardware), built-in landmark support (6 keypoints), consistent cross-platform behavior, and native ARM/Apple Silicon support. It represents the modern approach to face detection.

## 3. API Design

FastAPI was chosen over Flask for several reasons: native async support (critical for concurrent request handling), automatic OpenAPI documentation generation (valuable for evaluation), built-in request validation via Pydantic, and dependency injection support that maps cleanly to our detector registry pattern.

Key API design decisions:
- **Image input flexibility**: The `/detect` endpoint accepts both file uploads (`multipart/form-data`) and base64-encoded images (`application/json`), covering different client integration patterns.
- **Detector selection via query parameter**: `?detector=haar` rather than separate endpoints per detector. This keeps the API surface small and makes detector switching trivial for clients.
- **Blocking operations in thread pool**: Face detection is CPU-bound. Rather than blocking the async event loop, detection calls are dispatched via `asyncio.run_in_executor()` to a thread pool, allowing the server to handle concurrent requests even during long inference times.

## 4. Logging & Storage

Structured JSON logging was implemented with a dual-sink approach:
- **File sink**: JSON-formatted log files for operational monitoring, rotated by size.
- **SQLite sink**: Queryable storage for detection history, enabling analytics on detector usage, average processing times, and detection counts per user.

Each request is assigned a UUID (`X-Request-ID` header or auto-generated), enabling end-to-end tracing. User identification is passed via `X-User-ID` header — the service does not implement authentication, keeping concerns separated.

## 5. Concurrency & Thread Safety

The service is designed to run under `uvicorn` with multiple workers. Thread safety is ensured through:
- **No shared mutable state**: Each detector instance is either stateless or uses thread-local storage.
- **Detector instantiation**: Lazy-loaded singletons per worker process (not shared across processes).
- **SQLite access**: Write-ahead logging (WAL) mode enabled for concurrent read/write access.

## 6. Cross-Platform Strategy

All three detectors were selected partly for their cross-platform compatibility. Platform-specific considerations are handled at the dependency level:
- `dlib` requires `cmake` on macOS/Linux and Visual Studio Build Tools on Windows. This is documented in the README with platform-specific installation commands.
- The Dockerfile provides a platform-independent deployment path, eliminating local dependency issues entirely.
- No hardcoded file paths — all paths use `pathlib.Path` and are configurable via environment variables.
