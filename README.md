# Human_Face

Face detection, alignment, recognition, and similarity comparison service with image quality assessment.

## Architecture

```
┌─────────────┐
│  server.py  │  Flask HTTP API
└──────┬──────┘
       │
┌──────▼──────────────────────┐
│  inferencer/face_pipeline.py │  Business orchestration
└──────┬──────────────────────┘
       │
┌──────▼────────┐  ┌───────────────┐  ┌────────────────┐  ┌──────────────┐
│  face_detector │  │face_recognizer│  │ blur_detection │  │image_quality │  │dark_bg_detector│
│ (YOLO+dlib)  │  │   (TFace)     │  │   (ONNX)       │  │   (ONNX)     │  │(Model+CV)    │
└───────────────┘  └───────────────┘  └────────────────┘  └──────────────┘
```

## Project Structure

```
Human_Face/
├── server.py              # Flask API server
├── config.yaml            # Environment configuration (log path)
├── deploy.sh              # Deployment script (dev/stg/pro)
├── requirements.txt       # Python dependencies
├── version                # Code version identifier
├── inferencer/
│   ├── __init__.py
│   ├── face_pipeline.py   # Business orchestration + quality scoring
│   ├── face_detector.py   # YOLO detection + face validation + dlib alignment
│   ├── face_recognizer.py # TFace 512-d embedding extraction
│   ├── yolo_detection.py  # YOLO inference + NMS post-processing
│   ├── blur_detection.py  # Blur/sharpness classification
│   ├── image_quality.py   # Image quality (lighting) classification
│   └── dark_bg_detector.py # Dark background detection (two-stage)
├── utils/
│   └── log.py             # Custom rotating file logger
├── resources/             # Model weight files (.onnx, .dat)
├── docs/                  # Module documentation
├── unit_test/             # Integration test (single parameterized script)
└── log/                   # Log output directory
```

## Requirements

- Python 3.8.8
- CUDA 12.2

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

`config.yaml` at project root:

```yaml
log_path:
  dev: ./log/info_{hostname}.log
  stg: /mnt/pvc/1/log/hf_xy/info_{hostname}.log
  pro: /mnt/pvc/2/log/hf_xy/info_{hostname}.log

dark_bg:
  model_threshold: 0.95
  dark_pixel_thresh: 50
  dark_ratio_thresh: 0.75
  bright_pixel_thresh: 150
  bright_ratio_thresh: 0.1
  bbox_expand_ratio: 0.0
  bbox_expand_up_ratio: 0.1
```

The `log_path` key maps environment (`dev/stg/pro`) to log file paths. The active environment is selected by the `APP_ENV` environment variable (set by `deploy.sh`). `{hostname}` is automatically replaced with the sanitized machine hostname at runtime. `dark_bg.*` keys are optional; defaults shown above are used when omitted.

## Deployment

```bash
./deploy.sh {dev|stg|pro}
```

| Environment | Port  | Workers |
|-------------|-------|---------|
| dev         | 37709 | 2       |
| stg         | 80    | 4       |
| pro         | 80    | 6       |

## API Endpoints

### POST /face_register

Register a face with quality assessment.

**Request:**

| Field | Type | Description |
|-------|------|-------------|
| `recordID` | string | Request record ID |
| `sessionID` | string | Session ID |
| `msgID` | string | Message ID |
| `imgData` | string | Base64 encoded image |

**Response:**

| Field | Type | Description |
|-------|------|-------------|
| `recordID` | string | Echo request record ID |
| `sessionID` | string | Echo session ID |
| `msgID` | string | Echo message ID |
| `resp_code` | int | Status code (100=success, 200=blur, 201=backlight, 300=no face, 400=multi-face) |
| `imgQualityScore` | float | Composite image quality score [1-100] |
| `faceFeature` | string | Stringified 512-d face embedding vector |

### POST /head_detection

Real-time face detection and similarity comparison against a registered embedding.

**Request:**

| Field | Type | Description |
|-------|------|-------------|
| `recordID` | string | Request record ID |
| `sessionID` | string | Session ID |
| `msgID` | string | Message ID |
| `imgData` | string | Base64 encoded live image |
| `faceFeature` | string | Previously registered face embedding |

**Response:**

| Field | Type | Description |
|-------|------|-------------|
| `recordID` | string | Echo request record ID |
| `sessionID` | string | Echo session ID |
| `msgID` | string | Echo message ID |
| `resp_code` | int | Status code |
| `detectRes` | int | Number of faces detected |
| `top3Similarity` | list | Face detection confidence scores |
| `faceSimilarity` | list | Similarity score vs registered face |
| `msg` | string | User-facing status message |

### POST /dark_bg_check

Dark background detection using two-stage pipeline (quality model + CV background analysis).

**Request:**

| Field | Type | Description |
|-------|------|-------------|
| `recordID` | string | Request record ID |
| `sessionID` | string | Session ID |
| `msgID` | string | Message ID |
| `imgData` | string | Base64 encoded image |

**Response:**

| Field | Type | Description |
|-------|------|-------------|
| `recordID` | string | Echo request record ID |
| `sessionID` | string | Echo session ID |
| `msgID` | string | Echo message ID |
| `resp_code` | int | Status code (100=normal, 300=no face, 999=error) |
| `is_dark_bg` | bool | Whether the image has a dark background |
| `dark_score` | float | Stage 1 model class 2 score |
| `bg_ratio` | float | Stage 2 background dark pixel ratio |

### GET /version/

Returns server IP, start time, code version, and gunicorn worker count.

## Status Codes

| Code | Meaning |
|------|---------|
| 100 | Success |
| 200 | Image is blurry |
| 201 | Backlight detected |
| 221 | Blurry + backlight |
| 300 | No face detected |
| 400 | Multiple faces detected |
| 500 | No registered feature provided |
| 600 | Feature extraction error |
| 999 | Internal server error |

## Documentation

Detailed module documentation available in `docs/`:

- [server.py](docs/server.md)
- [api.md](docs/api.md) — API reference for upstream/downstream integration
- [face_pipeline.py](docs/inferencer/face_pipeline.md)
- [face_detector.py](docs/inferencer/face_detector.md)
- [face_recognizer.py](docs/inferencer/face_recognizer.md)
- [yolo_detection.py](docs/inferencer/yolo_detection.md)
- [blur_detection.py](docs/inferencer/blur_detection.md)
- [image_quality.py](docs/inferencer/image_quality.md)
- [dark_bg_detector.py](docs/inferencer/dark_bg_detector.md)
