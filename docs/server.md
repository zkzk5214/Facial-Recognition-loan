# server.py - Module Documentation

## 1. Module Overview

**Core Responsibility:** Provides the Flask HTTP API layer for face registration and head detection services, handling request parsing, response formatting, error handling, and logging while delegating core inference logic to the `face_pipeline` module.

**Key Functionalities:**
- `/face_register` (POST): Register a face — detect, validate quality, and extract face embedding
- `/head_detection` (POST): Detect face in real-time frame, compare against registered embedding
- `/version/` (GET): Health check endpoint returning server version and worker count
- Unified request parsing (base64 image decoding) and response formatting
- Structured logging with environment-configurable log path

---

## 2. Architecture & Design Patterns

### Dependencies

| Library | Purpose |
|---------|---------|
| `flask` | HTTP server framework (routes, request parsing, JSON response) |
| `werkzeug` | WSGI request handler configuration (HTTP/1.1) |
| `cv2` (OpenCV) | Base64 image decoding |
| `numpy` | Image buffer conversion, array operations |
| `yaml` | Load configuration from `config.yaml` |
| `utils.log` | Custom logger creation |
| `inferencer.face_pipeline` | Core inference pipeline (`face_register_`, `head_detection_`, `get_topk`, `StatusCode`) |

### Design Patterns Applied

- **Facade Pattern**: Flask routes act as thin facades over `face_pipeline` functions, handling only HTTP concerns (parsing, response formatting, error handling)
- **Early Return Pattern**: `head_detection` uses early returns to flatten nested error handling
- **Configuration Externalization**: Log path loaded from `config.yaml` for environment portability

### Data Flow

```
HTTP POST (JSON with base64 image)
    -> process_common_request: parse JSON + decode base64 -> np.ndarray
    -> face_register_ / head_detection_: core inference pipeline
    -> create_response: build response dict with IDs
    -> Populate response fields (features, quality score, similarity)
    -> log_request_result / log_warning: structured logging
    -> jsonify: return JSON response
```

---

## 3. Core Components Deep Dive

### `/face_register` (POST)

| Item | Detail |
|------|--------|
| **Purpose** | Register a face: detect, assess quality, extract 512-d embedding |
| **Request** | JSON with `recordID`, `sessionID`, `msgID`, `imgData` (base64 BGR image) |
| **Response** | `{resp_code, imgQualityScore, faceFeature}` — status code, quality [1-100], stringified feature vector |
| **Error Handling** | Returns `resp_code=999` on uncaught exceptions; logs warning if no feature extracted |

---

### `/head_detection` (POST)

| Item | Detail |
|------|--------|
| **Purpose** | Detect face in real-time frame and compute similarity against registered embedding |
| **Request** | JSON with `recordID`, `sessionID`, `msgID`, `imgData`, `faceFeature` (stringified registered vector) |
| **Response** | `{resp_code, detectRes, top3Similarity, faceSimilarity, msg}` — status, face count, confidences, similarity scores, user-facing message |
| **Flow** | Validate faceFeature exists → detect faces → check quality flags → extract embedding → compute cosine similarity via `get_topk` |

---

### Helper Functions

| Function | Purpose |
|----------|---------|
| `decode_base64_image` | Base64 string → BGR `np.ndarray` via `cv2.imdecode` |
| `process_common_request` | Parse Flask request body and decode image in one call |
| `create_response` | Build response dict with session IDs and default status code |
| `log_request_result` | Log successful request with timing and metadata |
| `log_warning` | Log warning events (no face, blur, etc.) with context |

---

## 4. Constraints, Edge Cases & Side Effects

### Preconditions

- `config.yaml` must exist at project root with `log_path` key
- `./version` file must exist (read at startup)
- All model weight files must be present (loaded transitively via `face_pipeline` import)
- Request JSON must contain `recordID`, `sessionID`, `msgID` fields

### Known Limitations

- **Single-threaded by default** (`threaded=False`): Production relies on gunicorn for concurrency
- **No request size limit**: Large base64 images could cause memory issues
- **`json.loads(json_data['faceFeature'])` parsing**: If client sends malformed feature string, will raise exception (caught by try/except)
- **`os.popen` in `/version/`**: Spawns a shell process on every call; not ideal for high-frequency health checks
- **Logging includes raw `imgData`** in warnings: Base64 image data in logs causes massive log file sizes

### Side Effect Warnings

- Module import triggers loading of 4 ML models via `face_pipeline` (GPU memory allocation, ~5-10s cold start)
- Logger writes to file system continuously
- `WSGIRequestHandler.protocol_version` globally modifies werkzeug behavior

---

## 5. Vibe Coding Extension & Maintenance Guide

### Safe Modification Zones

| Zone | Safety | Notes |
|------|--------|-------|
| Response field names/values | Moderate | Frontend depends on exact field names; coordinate with client team |
| Log format/content | Safe | Internal; can adjust without breaking API |
| `config.yaml` fields | Safe | Can add new config items without code changes |
| Status code messages (Chinese strings) | Safe | User-facing text; can localize |
| `/version/` endpoint | Safe | Monitoring only |

### Common Refactoring Pitfalls

- **Changing response field names**: Client applications depend on `resp_code`, `faceFeature`, `imgQualityScore`, `faceSimilarity`, etc. Any rename is a breaking API change
- **Modifying `StatusCode` values**: Must stay consistent between `server.py` and `face_pipeline.py`; also affects client-side status handling
- **Removing `try/except` blocks**: Routes must always return valid JSON; unhandled exceptions would return HTML error pages
- **Changing `faceFeature` serialization format**: Currently `str(list)` format; client uses `json.loads` to parse; changing format breaks compatibility

### Testing Recommendations

- **API contract test**: Send valid request to `/face_register`; verify response contains all expected fields with correct types
- **Error handling test**: Send request with missing `imgData`; verify `resp_code=999` and no server crash
- **No-face test**: Send image without faces; verify `resp_code=300`
- **Similarity test**: Register a face, then send same face to `/head_detection`; verify `faceSimilarity > 90`
- **Load test**: Verify gunicorn workers handle concurrent requests without model conflicts
