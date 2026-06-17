# face_pipeline.py - Module Documentation

## 1. Module Overview

**Core Responsibility:** Orchestrates the face processing pipeline — from image preprocessing, face detection and alignment, to feature extraction and similarity comparison — serving as the business logic layer between the Flask server and individual inference modules.

**Key Functionalities:**
- Face registration: detect, align, and extract face embeddings
- Top-K similarity search between registered and query face feature vectors
- Image preprocessing (downscale large images)

---

## 2. Architecture & Design Patterns

### Dependencies

| Library | Purpose |
|---------|---------|
| `cv2` (OpenCV) | Image resize |
| `numpy` | Array operations |
| `sklearn.metrics.pairwise` | `cosine_similarity` for face feature comparison |
| `inferencer.face_detector` | YOLO + dlib face detection and alignment |
| `inferencer.face_recognizer` | Face embedding extraction (TFace) |

### Design Patterns Applied

- **Facade Pattern**: `face_register` provides a single high-level API hiding the complexity of detection + alignment + recognition.
- **Module-level Singleton**: Models are instantiated once at module level and reused across all function calls.

### Data Flow

```
face_register(img):
    Input (BGR image)
        -> preprocess_image: downscale if height > 2400px
        -> face_detector.get_face_capture: detect + validate + align face chips
        -> face_recognizer.recognize: extract 512-d embedding per face
        -> Output: (features, face_conf) or features

get_topk(register, query):
    Input (two feature arrays)
        -> cosine_similarity: compute similarity matrix
        -> Apply margin penalty (-3)
        -> Sort descending, take top-K scores
        -> Output: List[int] (similarity percentages)
```

---

## 3. Core Components Deep Dive

### `preprocess_image(img)`

| Item | Detail |
|------|--------|
| **Purpose** | Downscale large images to limit detection input size |
| **Parameters** | `img`: `np.ndarray` (BGR, uint8) |
| **Return** | Resized image if height > 2400, otherwise original unchanged |
| **Core Logic** | Preserves aspect ratio; scales height to 2400 and width proportionally |

---

### `get_topk(register_features, query_features, topk=3)`

| Item | Detail |
|------|--------|
| **Purpose** | Find top-K similarity scores between registered and query face embeddings |
| **Parameters** | `register_features`: `np.ndarray` (M, 512); `query_features`: `np.ndarray` (N, 512); `topk`: `int` (default 3) |
| **Return** | `List[int]` — top-K similarity scores (0-100 range, with -3 margin penalty applied) |
| **Core Logic** | Computes cosine similarity matrix × 100, subtracts 3 as margin penalty, takes max across registered axis per query, sorts descending, returns top-K integer scores |

---

### `face_register(img, get_conf=True)`

| Item | Detail |
|------|--------|
| **Purpose** | Detect faces and extract feature embeddings from an image |
| **Parameters** | `img`: `np.ndarray` (BGR, uint8); `get_conf`: `bool` — whether to also return confidence scores |
| **Return** | If `get_conf=True`: `(features, face_conf)` or `([], [])` if no faces. If `get_conf=False`: `features` or `[]` if no faces |
| **Core Logic** | Preprocess → detect + align faces → extract 512-d embedding per face chip |

---

## 4. Constraints, Edge Cases & Side Effects

### Preconditions

- Model weight files must exist: `arcface_weights_best_new.onnx`, `shape_predictor_68_face_landmarks.dat`, `TFace_pretrain.onnx` under `./resources/`
- Input images must be valid BGR `np.ndarray` (uint8, 3 channels)
- `get_topk` requires both feature arrays to have the same embedding dimension (512)

### Known Limitations

- **Module-level model loading**: Importing this module triggers loading of 2 heavy models (YOLO+dlib, TFace). Slow cold start
- **No concurrency safety**: Module-level model instances are shared; concurrent calls from multiple threads may cause issues
- **`- 3` margin penalty in `get_topk`**: Hardcoded offset reduces all similarity scores by 3%; rationale undocumented
- **No quality assessment**: Current version does not check blur/lighting/exposure (previously available features have been removed)

### Side Effect Warnings

- Module import triggers GPU memory allocation for ONNX models
- No file I/O or global state mutations during inference
- `preprocess_image` does NOT modify the original image (returns a new array or the original unchanged)

---

## 5. Vibe Coding Extension & Maintenance Guide

### Safe Modification Zones

| Zone | Safety | Notes |
|------|--------|-------|
| `preprocess_image` size limit (2400) | Safe | Adjustable based on hardware capacity |
| `get_topk` margin value (-3) | Safe | Tunable; affects reported similarity |
| `get_topk` topk default (3) | Safe | Adjustable per use case |
| `RESOURCE_PATH` | Moderate | Must match actual model file locations |
| `face_register` return format | Caution | `server.py` depends on exact tuple structure |

### Common Refactoring Pitfalls

- **Changing `face_register` return format**: `server.py` destructures as `(features, conf)` or expects a list — any change breaks the API
- **Renaming exported functions**: `server.py` imports `face_register`, `get_topk` by name
- **Moving model initialization into a function**: `face_register` references module-level `face_detector` and `face_recognizer`; would require dependency injection or class refactor
- **Adding quality checks back**: Would require re-importing `blur_detection`, `image_quality` and updating return format

### Testing Recommendations

- **No-face test**: Provide a landscape image; verify `([], [])` returned
- **Single face test**: Provide a clear frontal face; verify feature vector shape is (512,)
- **Multi-face test**: Provide image with 2+ faces; verify correct count of features and confidences
- **Similarity test**: Register a face, then query with same face via `get_topk`; verify score > 90
- **get_conf=False test**: Verify returns flat list (not tuple) when `get_conf=False`
