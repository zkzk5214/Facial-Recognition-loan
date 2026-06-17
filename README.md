# Human_Face

Face detection, alignment, recognition, and similarity comparison service.

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
┌──────▼────────┐  ┌───────────────┐  ┌──────────────┐
│ face_detector │  │face_recognizer│  │ mask_detect  │
│ (YOLO+dlib)  │  │   (TFace)     │  │  (ONNX CNN) │
└───────────────┘  └───────────────┘  └──────────────┘
```

## Project Structure

```
Human_Face/
├── server.py              # Flask API server
├── config.yaml            # Environment configuration
├── deploy.sh              # Deployment script (dev/stg/pro)
├── version                # Code version identifier
├── inferencer/
│   ├── __init__.py
│   ├── face_pipeline.py   # Business orchestration layer
│   ├── face_detector.py   # YOLO detection + validation + dlib alignment
│   ├── face_recognizer.py # TFace embedding extraction
│   ├── yolo_detection.py  # YOLO inference + NMS post-processing
│   └── mask_detect.py     # Mask/no-mask classification
├── utils/
│   └── log.py             # Custom rotating file logger
├── resources/             # Model weight files
├── docs/                  # Module documentation
├── unit_test/             # Integration tests
└── log/                   # Log output directory
```

## Requirements

- Python 3.8.8
- CUDA 12.2


## Configuration

`config.yaml` at project root:

```yaml
log_path: /path/to/log/info_{hostname}.log
```

`{hostname}` is automatically replaced with the sanitized machine hostname at runtime.

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

### POST /head_detection

Full face comparison pipeline with DS + CS registered images.

**Request:**

| Field | Type | Description |
|-------|------|-------------|
| `recordID` | string | Request record ID |
| `msgID` | string | Message ID |
| `imgDataRegisterDS` | string | Base64 encoded DS registration image |
| `imgDataRegisterCS` | string | Base64 encoded CS registration image |
| `imgData` | string | Base64 encoded live group image |

**Response:**

| Field | Type | Description |
|-------|------|-------------|
| `recordID` | string | Echo request record ID |
| `msgID` | string | Echo message ID |
| `detectionSimilarity` | list | Face detection confidence scores |
| `faceSimilarityDS` | list | Top-K similarity vs DS registration |
| `faceSimilarityCS` | list | Top-K similarity vs CS registration |
| `imgQuality` | object | `{size, blur, bright, dark}` quality flags (1=pass, 0=fail) |
| `mask` | int | 0 = mask detected, 1 = no mask |

### POST /ds_head_detection

Simplified DS-only face comparison.

**Request:**

| Field | Type | Description |
|-------|------|-------------|
| `msgID` | string | Message ID |
| `imgDataRegisterDS` | string | Base64 encoded DS registration image |
| `imgData` | string | Base64 encoded live group image |

**Response:**

| Field | Type | Description |
|-------|------|-------------|
| `msgID` | string | Echo message ID |
| `detectionSimilarity` | list | Face detection confidence scores |
| `faceSimilarityDS` | list | Top-K similarity vs DS registration |
| `imgQuality` | object | `{size, blur, bright, dark}` quality flags |

### GET /version/

Returns server IP, start time, code version, and gunicorn worker count.

## Documentation

Detailed module documentation available in `docs/`:

- [server.py](docs/server.md)
- [face_pipeline.py](docs/inferencer/face_pipeline.md)
- [face_detector.py](docs/inferencer/face_detector.md)
- [face_recognizer.py](docs/inferencer/face_recognizer.md)
- [yolo_detection.py](docs/inferencer/yolo_detection.md)
- [mask_detect.py](docs/inferencer/mask_detect.md)
