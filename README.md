# Face Detection Service

Modular, production-ready face detection API supporting three detection algorithms behind a unified interface.

## Detectors

| Detector | Key | Description |
|----------|-----|-------------|
| **Haar Cascade** | `haar` | OpenCV Viola-Jones classifier — fast, zero extra dependencies |
| **Dlib HOG** | `dlib_hog` | Histogram of Oriented Gradients + SVM — balanced speed/accuracy |
| **MediaPipe** | `mediapipe` | Google BlazeFace neural network — modern, includes 6-point landmarks |

## Installation

### Prerequisites
- Python 3.10+
- cmake (required for dlib)

### macOS
```bash
brew install cmake
python -m venv venv
source venv/bin/activate
pip install -r face_detection_service/requirements.txt
```

### Linux (Ubuntu/Debian)
```bash
sudo apt-get install cmake build-essential
python3 -m venv venv
source venv/bin/activate
pip install -r face_detection_service/requirements.txt
```

### Windows
```powershell
# Install Visual Studio Build Tools with C++ workload first
python -m venv venv
venv\Scripts\activate
pip install -r face_detection_service/requirements.txt
```

## Running

### Local
```bash
python -m face_detection_service.main
# Server starts at http://localhost:8000
```

### Docker
```bash
docker compose up --build
```

## API Usage

### Detect faces (file upload)
```bash
curl -X POST http://localhost:8000/detect \
  -F "file=@photo.jpg" \
  -G -d "detector=haar"
```

### Detect faces (base64)
```bash
curl -X POST "http://localhost:8000/detect?detector=mediapipe" \
  -H "Content-Type: application/json" \
  -d '{"image_base64": "<base64-data>"}'
```

### Detect with annotated image
```bash
curl -X POST http://localhost:8000/detect \
  -F "file=@photo.jpg" \
  -G -d "detector=dlib_hog" -d "annotated=true"
```

### With request tracing headers
```bash
curl -X POST http://localhost:8000/detect \
  -F "file=@photo.jpg" \
  -H "X-Request-ID: my-request-123" \
  -H "X-User-ID: user-abc"
```

### Health check
```bash
curl http://localhost:8000/health
```

**Response:**
```json
{"status": "ok", "available_detectors": ["haar", "dlib_hog", "mediapipe"]}
```

### List detectors
```bash
curl http://localhost:8000/detectors
```

### Example detection response
```json
{
  "detections": [
    {
      "bbox": [244, 282, 680, 680],
      "confidence": 0.9568,
      "landmarks": {
        "right_eye": [386, 495],
        "left_eye": [662, 496],
        "nose_tip": [476, 683],
        "mouth_center": [502, 802],
        "right_ear_tragion": [319, 533],
        "left_ear_tragion": [882, 530]
      },
      "metadata": {}
    }
  ],
  "num_faces": 1,
  "detector_used": "mediapipe",
  "processing_time_ms": 14.89,
  "image_metadata": {"width": 1024, "height": 1024, "channels": 3, "format": null},
  "annotated_image_base64": null
}
```

## Switching Detectors

Pass the `detector` query parameter to `/detect`:

```bash
# Haar Cascade (fastest, classical)
curl -X POST http://localhost:8000/detect -F "file=@photo.jpg" -G -d "detector=haar"

# Dlib HOG (balanced)
curl -X POST http://localhost:8000/detect -F "file=@photo.jpg" -G -d "detector=dlib_hog"

# MediaPipe (most accurate, with landmarks)
curl -X POST http://localhost:8000/detect -F "file=@photo.jpg" -G -d "detector=mediapipe"
```

## Running Tests

```bash
pip install -r face_detection_service/requirements.txt
pytest
```

## Benchmark

```bash
python -m face_detection_service.benchmarks.run_benchmark
```

Compares all available detectors across small (320x240), medium (640x480), and large (1280x960) images using a real face photo downloaded at runtime. Reports min/median/max inference time, face count, and average confidence.

### Example benchmark output (Apple M3 Pro)

| Detector   | Image            | Min ms | Median ms | Max ms | Faces | Avg conf |
|------------|------------------|--------|-----------|--------|-------|----------|
| haar       | small (320x240)  |   3.08 |      3.09 |  39.95 |     1 |    1.000 |
| dlib_hog   | small (320x240)  |  16.80 |     16.88 |  17.35 |     1 |    0.820 |
| mediapipe  | small (320x240)  |   1.16 |      1.20 |   2.18 |     1 |    0.909 |
| haar       | medium (640x480) |  10.29 |     10.93 |  12.61 |     1 |    1.000 |
| dlib_hog   | medium (640x480) |  67.27 |     67.47 |  68.05 |     1 |    0.763 |
| mediapipe  | medium (640x480) |   1.20 |      1.23 |   1.50 |     1 |    0.911 |
| haar       | large (1280x960) |  34.41 |     35.35 |  38.78 |     2 |    0.750 |
| dlib_hog   | large (1280x960) | 262.15 |    263.85 | 268.39 |     1 |    0.779 |
| mediapipe  | large (1280x960) |   1.33 |      1.39 |   1.79 |     1 |    0.905 |

## Interactive API Docs

Once the server is running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Configuration

All settings are configurable via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8000` | Server port |
| `WORKERS` | `1` | Uvicorn worker count |
| `LOG_LEVEL` | `INFO` | Logging level |
| `MAX_IMAGE_SIZE_MB` | `10` | Max image size in MB |
| `MAX_IMAGE_DIMENSION` | `4096` | Max image width/height |
| `HAAR_SCALE_FACTOR` | `1.1` | Haar cascade scale factor |
| `HAAR_MIN_NEIGHBORS` | `5` | Haar cascade min neighbors |
| `DLIB_UPSAMPLE_NUM_TIMES` | `1` | Dlib HOG upsample count |
| `MEDIAPIPE_MIN_DETECTION_CONFIDENCE` | `0.5` | MediaPipe min confidence |
| `MEDIAPIPE_MODEL_SELECTION` | `0` | 0=short-range, 1=full-range |
| `THREAD_POOL_MAX_WORKERS` | `4` | Thread pool size for detection |

## Assumptions & Limitations

- **No authentication**: User identification via `X-User-ID` header; no auth layer included.
- **Haar confidence**: Uses `detectMultiScale3` reject-level weights normalised to [0, 1]; falls back to 1.0 if unavailable.
- **Synthetic test images**: Unit tests use generated images; real-world accuracy requires actual face photos.
- **dlib installation**: Requires cmake and a C++ compiler; Docker eliminates this dependency issue.
- **Single-node**: Designed for single-node deployment; horizontal scaling would require shared storage for SQLite logs.
