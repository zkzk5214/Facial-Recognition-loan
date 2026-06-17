# server.py - Module Documentation

## 1. Module Overview

**Core Responsibility:** Provides the Flask HTTP API layer for face detection and similarity comparison services, handling base64 image decoding, mask detection, face feature extraction, and Top-K similarity scoring.

**Key Functionalities:**
- `/head_detection` (POST): Full pipeline — decode DS/CS/Group images, mask detection, face registration, DS+CS similarity comparison
- `/ds_head_detection` (POST): Simplified pipeline — decode DS/Group images, face registration, DS similarity comparison only
- `/version/` (GET): Health check endpoint returning server version and gunicorn worker count
- Base64 image decoding utility

---

## 2. Architecture & Design Patterns

### Dependencies

| Library | Purpose |
|---------|---------|
| `flask` | HTTP server framework (routes, request parsing, JSON response) |
| `werkzeug` | WSGI request handler configuration (HTTP/1.1) |
| `cv2` (OpenCV) | Base64 image decoding via `imdecode` |
| `numpy` | Image buffer conversion (`frombuffer`) |
| `yaml` | Load log path configuration from `config.yaml` |
| `utils.log` | Custom logger with daily rotation |
| `inferencer.mask_detect` | Mask/no-mask classification |
| `inferencer.face_pipeline` | `face_register` (face detection + embedding), `get_topk` (similarity) |

### Design Patterns Applied

- **Facade Pattern**: Flask routes act as thin facades over inference pipeline functions, handling only HTTP concerns.
- **Early Return Pattern**: Both routes use early returns for validation failures (image too small, no face detected).
- **Configuration Externalization**: Log path loaded from `config.yaml` for environment portability.

### Data Flow

```
/head_detection:
    HTTP POST (JSON with imgDataRegisterDS, imgDataRegisterCS, imgData)
        -> decode_base64_image × 3
        -> mask_detect.get_class(imgData) → mask flag
        -> face_register(DS, get_conf=False) → ds_feature
        -> face_register(CS, get_conf=False) → cs_feature
        -> face_register(Group) → group_feature + detectionConf
        -> get_topk(ds vs group) → faceSimilarityDS
        -> get_topk(cs vs group) → faceSimilarityCS
        -> jsonify response

/ds_head_detection:
    HTTP POST (JSON with imgDataRegisterDS, imgData)
        -> decode_base64_image × 2
        -> face_register(DS, get_conf=False) → ds_feature
        -> face_register(Group) → img_feature + detectionConf
        -> get_topk(ds vs group) → faceSimilarityDS
        -> jsonify response
```

---

## 3. Core Components Deep Dive

### `/head_detection` (POST)

| Item | Detail |
|------|--------|
| **Purpose** | Full face comparison: detect faces in 3 images (DS register, CS register, live group), compute mask status and similarity scores |
| **Request** | JSON with `recordID`, `msgID`, `imgDataRegisterDS`, `imgDataRegisterCS`, `imgData` (all base64) |
| **Response** | `{recordID, msgID, detectionSimilarity, faceSimilarityDS, faceSimilarityCS, imgQuality, mask}` |
| **Early Exits** | Image base64 too short (< 1000 chars) → size error; no group faces detected → incomplete |
| **Error Handling** | Returns hardcoded response with `[100, 100]` scores on uncaught exceptions |

---

### `/ds_head_detection` (POST)

| Item | Detail |
|------|--------|
| **Purpose** | Simplified DS-only comparison: detect faces in 2 images (DS register, live group), compute DS similarity |
| **Request** | JSON with `msgID`, `imgDataRegisterDS`, `imgData` (base64) |
| **Response** | `{msgID, detectionSimilarity, faceSimilarityDS, imgQuality}` |
| **Early Exits** | Image too short; no group faces; no DS faces |
| **Error Handling** | Returns hardcoded response with `[100]` scores on uncaught exceptions |

---

### `/version/` (GET)

| Item | Detail |
|------|--------|
| **Purpose** | Health check returning server IP, start time, code version, and gunicorn worker count |
| **Response** | Plain text string |

---

### Helper Function

| Function | Purpose |
|----------|---------|
| `decode_base64_image(base64_data)` | Base64 string → BGR `np.ndarray` via `np.frombuffer` + `cv2.imdecode` |

---

## 4. Constraints, Edge Cases & Side Effects

### Preconditions

- `config.yaml` must exist at project root with `log_path` key
- `./version` file must exist (read at startup)
- Model weight files must be present (loaded transitively via `face_pipeline` and `mask_detect` imports)
- Request JSON must contain `msgID` and image data fields

### Known Limitations

- **Single-threaded by default** (`threaded=False`): Production relies on gunicorn for concurrency
- **No request size limit**: Large base64 images could cause memory issues
- **Exception returns fake scores**: Both routes return `[100]` on error, making errors look like perfect matches
- **`except` block re-parses request**: If the original parse failed, the `except` block will also fail (secondary crash)
- **`os.popen` in `/version/`**: Spawns a shell process on every call
- **Similarity only computed for single-face registrations**: `if len(ds_feature) == 1` skips multi-face DS images silently

### Side Effect Warnings

- Module import triggers loading of ML models via `face_pipeline` and `mask_detect` (GPU memory allocation)
- Logger writes to file system continuously
- `WSGIRequestHandler.protocol_version` globally modifies werkzeug behavior

---

## 5. Vibe Coding Extension & Maintenance Guide

### Safe Modification Zones

| Zone | Safety | Notes |
|------|--------|-------|
| Response field values | Moderate | Frontend depends on exact field names |
| Log format/content | Safe | Internal; can adjust without breaking API |
| `config.yaml` fields | Safe | Can add new config items |
| `/version/` endpoint | Safe | Monitoring only |
| Image size threshold (1000) | Safe | Tunable validation |

### Common Refactoring Pitfalls

- **Changing response field names**: Client applications depend on `detectionSimilarity`, `faceSimilarityDS`, `faceSimilarityCS`, `imgQuality`, `mask`
- **Removing `try/except` blocks**: Routes must always return valid JSON; unhandled exceptions return HTML error pages
- **Changing `face_register` return format**: Both routes destructure as `(features, conf)` or plain list depending on `get_conf`
- **Modifying mask detection logic**: `0 if get_class(img) else 1` — inverting breaks client interpretation

### Testing Recommendations

- **API contract test**: Send valid request to `/head_detection`; verify response contains all expected fields
- **Error handling test**: Send malformed JSON; verify no server crash and response is returned
- **No-face test**: Send image without faces; verify `detectionSimilarity` is empty
- **Mask test**: Send image with masked face; verify `mask` field is 0
- **DS similarity test**: Register face via DS, send same face as group; verify `faceSimilarityDS` score > 90
- **Image size test**: Send base64 string < 1000 chars; verify `imgQuality.size` is 0
