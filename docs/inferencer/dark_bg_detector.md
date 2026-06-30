# dark_bg_detector.py - Module Documentation

## 1. Module Overview

**Core Responsibility:** Provides a two-stage dark background detection pipeline — Stage 1 uses an ONNX image quality model for coarse filtering, Stage 2 uses CV-based background dark pixel ratio analysis for precise judgment.

**Key Functionalities:**
- Stage 1: Model inference via `ImgQuality` (class 2 score ≥ `model_threshold`)
- Stage 2: CV-based background dark pixel ratio analysis using face bounding box
- Unified `dark_bg_check()` function returning boolean result with debug scores
- Lazy initialization of independent model instances (not shared with `face_pipeline`)
- Configuration-driven thresholds via `config.yaml` (`dark_bg.*`)
- Standalone self-test via `if __name__ == '__main__'` (uses `sys.path` injection to resolve package imports)

---

## 2. Architecture & Design Patterns

### Dependencies

| Library | Purpose |
|---------|---------|
| `cv2` (OpenCV) | Grayscale conversion (BGR2GRAY), image analysis |
| `numpy` | Array operations, mask creation, pixel statistics |
| `yaml` | Load `dark_bg.*` configuration from `config.yaml` |
| `inferencer.image_quality.ImgQuality` | Stage 1 ONNX model inference (independent instance) |
| `inferencer.face_detector.Detector` | Stage 2 face bounding box detection (independent instance, `get_face_bboxes` only) |
| `inferencer.face_pipeline.StatusCode` | `StatusCode.NOFACE` (300) for no-face response |

### Design Patterns Applied

- **Lazy Initialization**: `_init()` singleton-pattern — models are loaded once on first call, not at module import time
- **Pipeline Pattern**: Sequential two-stage processing with early exits at each stage
- **Configuration Externalization**: All thresholds loaded from `config.yaml` (`dark_bg.*` namespace)

### Data Flow

```
Input (np.ndarray BGR image)
    │
    ├─ Stage 1: _quality_detector.detect(img) → 4-class scores
    │     ├─ argmax ≠ 2 or dark_score < 0.95 → return (False, 100, dark_score, 0.0)
    │     └─ pass → proceed to Stage 2
    │
    └─ Stage 2: _face_detector.get_face_bboxes(img)
          ├─ no faces → return (False, 300, dark_score, 0.0)
          └─ has faces → detect_bg_darkness(img, face_bbox)
                ├─ bg_pixels == 0 → return (False, -1.0) [sentinel: no bg → triggers fallback to stage1]
                ├─ bg_pixels > 0, dark_ratio == 0.0 → return (False, 0.0) [bg exists but no dark pixels]
                └─ bg_pixels > 0, dark_ratio > 0.0 → return (is_dark, 100, dark_score, bg_ratio)

Output: (is_dark_bg: bool, resp_code: int, dark_score: float, bg_ratio: float)
```

---

## 3. Core Components Deep Dive

### `_init()`

| Item | Detail |
|------|--------|
| **Purpose** | Lazily initialize independent `ImgQuality` and `Detector` instances |
| **Parameters** | None |
| **Return** | None (sets module globals `_quality_detector`, `_face_detector`) |
| **Core Logic** | Checks if instances are `None`; if so, creates `ImgQuality` with `weights-finetune-1-40--A.onnx` and `Detector` with YOLO model + dlib landmark file. Does NOT use `face_pipeline` module-level singletons |

---

### `detect_bg_darkness(img_bgr, face_bbox)`

| Item | Detail |
|------|--------|
| **Purpose** | CV-based dark pixel ratio analysis on background region |
| **Parameters** | `img_bgr`: `np.ndarray` (BGR, HWC, uint8); `face_bbox`: `[x1, y1, x2, y2]` (int) |
| **Return** | `(bool, float)` — whether background is dark, and the dark pixel ratio. Returns `(False, -1.0)` as sentinel when no background pixels exist (face fills image). Returns `(False, 0.0)` when background exists but has zero dark pixels. |
| **Core Logic** | 1. Convert to grayscale (`cv2.COLOR_BGR2GRAY`, BT.601); 2. Expand bbox (up: 30%, other: 20% of bbox size); 3. Clip expanded bbox to image boundaries; 4. Create mask: face = 0, bg = 255; 5. Count dark pixels (gray < `dark_pixel_thresh`) in bg region; 6. If `dark_ratio > dark_ratio_thresh` → dark background |

---

### `dark_bg_check(img_bgr)`

| Item | Detail |
|------|--------|
| **Purpose** | Unified two-stage dark background detection entry point |
| **Parameters** | `img_bgr`: `np.ndarray` (BGR) — full input image |
| **Return** | `(is_dark_bg, resp_code, dark_score, bg_ratio)` — final judgment with debug scores |
| **Core Logic** | Stage 1: model inference → early exit if not dark class or score < threshold; Stage 2: detect face bbox → no face → resp=300; has face → CV analysis; if bg ratio == -1.0 (sentinel, no bg pixels) → fallback to stage1 result; otherwise (bg ratio >= 0) → stage2 result |

---

## 4. Constraints, Edge Cases & Side Effects

### Preconditions

- `weights-finetune-1-40--A.onnx` must exist at `./resources/` (for `ImgQuality`)
- `arcface_weights_best_new.onnx` and `shape_predictor_68_face_landmarks.dat` must exist at `./resources/` (for `Detector`)
- `config.yaml` must contain `dark_bg` section with all required keys (falls back to hardcoded defaults if missing or partial)
- Input image must be BGR `np.ndarray` (uint8, 3 channels)

### Known Limitations

| Limitation | Detail |
|------------|--------|
| numpy `bool_` | `(bg_pixels < thresh).mean()` returns a `numpy.float64`, and the comparison `> _dark_ratio_thresh` yields a `numpy.bool_` which is not JSON-serializable. Explicitly cast with `bool()` before returning to the API layer (`server.py` line 188 uses `jsonify`). |
| Stage 1 model specificity | `weights-finetune-1-40--A.onnx` is an internal model; availability tied to this project |
| dlib dependency | `Detector` instantiation loads dlib shape predictor even though `get_face_bboxes` does not use it; pure overhead for this module |
| Single-face assumption | Stage 2 uses only the first detected face bbox; multiple faces are ignored |
| Lazy init non-thread-safe | `_init()` is not thread-safe; concurrent first calls may create multiple instances |

### Side Effect Warnings

- Model loading allocates GPU memory (if CUDA available) that persists for process lifetime
- Module reads `config.yaml` at import time (file system I/O)
- No writes to file system during inference

---

## 5. Vibe Coding Extension & Maintenance Guide

### Safe Modification Zones

| Zone | Safety | Notes |
|------|--------|-------|
| Config thresholds in `config.yaml` | Safe | `dark_bg.*` values can be tuned without code changes |
| `detect_bg_darkness` expand ratios | Safe | Can adjust up/down expand ratios independently |
| `detect_bg_darkness` dark pixel definition | Safe | Can switch to HSV value-channel or add color bias |
| Stage 1 model path | Moderate | Must match model input/output spec (320×240, 4-class) |
| `dark_bg_check` return tuple format | Caution | `server.py` expects `(bool, int, float, float)` |

### Common Refactoring Pitfalls

- **Sharing `face_pipeline` singletons**: `dark_bg_detector` intentionally creates independent instances; sharing would couple two unrelated modules
- **Adding `/255` normalization to Stage 1**: The model expects raw `[0, 255]` float32 input; normalizing would break inference
- **Removing lazy init**: Moving model loading to module import would slow down all server startup (even for calls that never use `/dark_bg_check`)
- **Changing `StatusCode.NOFACE` import**: If the import is removed and 300 is hardcoded, it becomes decoupled from the project's status code convention

### Testing Recommendations

- **Self-test via `__main__`**: Run `python inferencer/dark_bg_detector.py` to test with `./unit_test/test_img/black_bg_test.jpg` directly (requires model files in `./resources/`).
- **Dark background image**: Verify `is_dark_bg=True`, `dark_score ≥ 0.95`, `bg_ratio > 0.6`
- **Normal lighting image**: Verify `is_dark_bg=False` (should exit at Stage 1)
- **No face image**: Verify `resp_code=300`, `is_dark_bg=False`
- **Face-filling image (no background)**: Verify fallback to Stage 1 result when `bg_ratio=0.0`
- **Config hot-reload test**: Change `dark_bg.dark_ratio_thresh` in yaml and verify new threshold takes effect on restart
