# face_pipeline.py - Module Documentation

## 1. Module Overview

**Core Responsibility:** Orchestrates the complete face processing pipeline — from image preprocessing, blur/quality detection, face detection and alignment, to feature extraction and similarity comparison — serving as the business logic layer between the Flask server and individual inference modules.

**Key Functionalities:**
- Face registration: detect, validate quality, align, and extract face embedding for enrollment
- Head detection: detect faces, assess quality, extract embeddings for real-time comparison
- Image quality scoring: composite score from blur, total variation, exposure, and lighting analysis
- Top-K similarity search between registered and query face feature vectors

---

## 2. Architecture & Design Patterns

### Dependencies

| Library | Purpose |
|---------|---------|
| `cv2` (OpenCV) | Image resize and brightness calculation |
| `numpy` | Numerical operations, argmax, array manipulation |
| `torchvision.transforms` | `ToTensor` for total variation input |
| `piq` | `total_variation` for image sharpness metric |
| `sklearn.metrics.pairwise` | `cosine_similarity` for face feature comparison |
| `inferencer.face_detector` | YOLO + dlib face detection and alignment |
| `inferencer.face_recognizer` | Face embedding extraction (TFace) |
| `inferencer.blur_detection` | Blur classification model |
| `inferencer.image_quality` | Image quality classification model |

### Design Patterns Applied

- **Facade Pattern**: `face_register_` and `head_detection_` provide high-level APIs that hide the complexity of multiple inference models and quality checks.
- **Strategy Pattern (implicit)**: `_determine_status` maps boolean quality flags to status codes using priority-based evaluation.
- **Module-level Singleton**: Models are instantiated once at module level and reused across all function calls.

### Data Flow

```
face_register_:
    Input (BGR image)
        -> preprocess_image: downscale if > 2400px
        -> face_detector: detect + align face chips
        -> Early exit if 0 or >1 faces
        -> detect_blur: blur probability check
        -> total_variation: texture complexity score
        -> exposure_loss: brightness deviation score
        -> detect_quality: lighting quality check on face chip
        -> Composite image_quality_score calculation
        -> _determine_status: map flags to StatusCode
        -> face_recognizer: extract 512-d embedding
        -> Output: (features, status_code, quality_score)

head_detection_:
    Input (BGR image)
        -> preprocess_image + face_detector
        -> Early exit if 0 or >1 faces
        -> detect_blur + detect_quality
        -> _determine_status
        -> face_recognizer: extract embeddings
        -> Output: (face_count, face_conf, features, status_code)
```

---

## 3. Core Components Deep Dive

### `face_register_(img)`

| Item | Detail |
|------|--------|
| **Purpose** | Full face registration pipeline: detect, validate, and extract face embedding |
| **Parameters** | `img`: `np.ndarray` (BGR, uint8) — raw input image |
| **Return** | `(face_features, status_code, image_quality_score)` — list of 512-d embeddings, int status code, float quality score [1, 100] |
| **Core Logic** | Preprocess → detect faces → early return for 0/multi face → assess blur + exposure + quality → compute composite quality score → determine status → extract embedding |

---

### `head_detection_(img)`

| Item | Detail |
|------|--------|
| **Purpose** | Real-time face detection for liveness/verification: detect, assess quality, extract embedding |
| **Parameters** | `img`: `np.ndarray` (BGR, uint8) |
| **Return** | `(face_count, face_conf, face_features, status_code)` — count, confidence list, embeddings, int status code |
| **Core Logic** | Lighter than `face_register_` — skips quality scoring, only checks blur/quality flags for status determination |

---

### `get_topk(register_features, query_features, topk=3)`

| Item | Detail |
|------|--------|
| **Purpose** | Find top-K most similar faces between registered and query feature sets |
| **Parameters** | `register_features`: `np.ndarray` (M, 512) registered embeddings; `query_features`: `np.ndarray` (N, 512) query embeddings; `topk`: `int` |
| **Return** | `(max_score, max_location, vectors)` — top-K similarity percentages (0-100), indices, and corresponding feature vectors |
| **Core Logic** | Computes cosine similarity matrix, takes max across registered axis, sorts descending, returns top-K |

---

### `_determine_status(is_blurry, is_low_quality)`

| Item | Detail |
|------|--------|
| **Purpose** | Map quality boolean flags to a prioritized StatusCode |
| **Parameters** | `is_blurry`: `bool`; `is_low_quality`: `bool` |
| **Return** | `StatusCode` enum member |
| **Core Logic** | Priority: BLURRED_BACKLIGHT (both) > BLURRED > BACKLIGHT > SUCCESS |

---

### `calculate_image_brightness(img_bgr)` / `exposure_loss(img_bgr)`

| Item | Detail |
|------|--------|
| **Purpose** | Compute perceived brightness (Rec. 709 luminance) and deviation from ideal exposure |
| **Parameters** | `img_bgr`: BGR image |
| **Return** | `brightness`: float [0, 255]; `exposure_loss`: float [0, 100] (0 = ideal, 100 = extreme over/underexposure) |
| **Core Logic** | Weighted sum of squared channel means (Rec. 709 coefficients), then measures squared distance from midpoint (127.5), normalized to [0, 100] |

---

## 4. Constraints, Edge Cases & Side Effects

### Preconditions

- All 4 model weight files must exist under `./resources/`
- Input images must be valid BGR `np.ndarray` (uint8, 3 channels)
- `get_topk` requires `register_features` and `query_features` to have compatible shapes for cosine similarity (same embedding dimension)

### Known Limitations

- **Module-level model loading**: Importing this module triggers loading of 4 heavy models (YOLO, dlib, TFace, blur, quality). Slow cold start (~5-10s)
- **No concurrency safety**: Module-level model instances are shared; concurrent calls from multiple threads may cause issues with ONNX Runtime sessions
- **Quality score formula is heuristic**: The composite `image_quality_score` combines multiple metrics with hardcoded weights (e.g., `* 0.1` for exposure); not calibrated against ground truth
- **`total_variation` requires torch**: Converts image to tensor just for one metric; adds latency
- **`head_detection_` returns different tuple shapes**: 0 faces → `(0, [], [], code)`, 1 face → `(1, conf, features, code)`, multi → `(count, conf, features, code)` — caller must handle all cases

### Side Effect Warnings

- Module import triggers GPU memory allocation for 4 ONNX models
- `sys.path.append` permanently mutates `sys.path`
- No file I/O or database writes during inference
- `preprocess_image` does NOT modify the original image (returns a new array or the original unchanged)

---

## 5. Vibe Coding Extension & Maintenance Guide

### Safe Modification Zones

| Zone | Safety | Notes |
|------|--------|-------|
| `StatusCode` enum values | Safe | Can add new codes; existing values are used by `server.py` |
| Quality score formula (lines 114-116) | Safe | Heuristic; tunable weights and thresholds |
| `exposure_loss` parameters (`mu`, scaling) | Safe | Adjustable for different camera conditions |
| `detect_blur` threshold default (0.75) | Safe | Tunable per deployment |
| `preprocess_image` size limit (2400) | Safe | Adjustable based on hardware capacity |
| `RESOURCE_PATH` | Moderate | Must match actual model file locations |
| Function return formats | Caution | `server.py` depends on exact tuple structures |

### Common Refactoring Pitfalls

- **Changing `face_register_` return format**: `server.py` destructures as `(img_feature, resp_flag, imgQualityScore)` — any change breaks the API
- **Renaming exported functions**: `server.py` imports `face_register_`, `get_topk`, `head_detection_` by name
- **Moving model initialization**: Functions reference module-level globals (`blur_detector`, etc.); moving init to a function requires passing instances or using a class
- **Changing `detect_blur` return semantics**: Both `face_register_` and `head_detection_` depend on truthy = blurry

### Testing Recommendations

- **Integration test**: Provide a known good image; verify `StatusCode.SUCCESS` and quality score > 80
- **Blur test**: Provide a motion-blurred image; verify `StatusCode.BLURRED` is returned
- **Multi-face test**: Provide an image with 2+ faces; verify `StatusCode.MULTIFACE` and correct feature count
- **No-face test**: Provide a landscape image; verify `StatusCode.NOFACE` and empty feature list
- **Similarity test**: Register a face, then query with same face; verify `get_topk` returns score > 90
