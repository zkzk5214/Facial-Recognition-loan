# dark_bg_detector.py - Module Documentation

## 1. Module Overview

**Core Responsibility:** Provides a two-stage dark background detection pipeline — Stage 1 uses an ONNX image quality model for coarse filtering, Stage 2 uses CV-based connected components analysis to distinguish "dark bg + isolated specular reflections" from "dark bg + large bright/gray regions (glass, seats)".

**Key Functionalities:**
- Stage 1: Model inference via `ImgQuality` (class 2 score ≥ `model_threshold`)
- Stage 2: Connected components analysis on background pixels — separates small specular spots (`spot_ratio`) from large bright patches (`patch_ratio`) by area threshold
- Unified `dark_bg_check()` function returning boolean result with debug scores
- Shared model instances from `face_pipeline` (no duplicate ONNX sessions)
- Configuration-driven thresholds via `config.yaml` (`dark_bg.*`)

---

## 2. Architecture & Design Patterns

### Dependencies

| Library | Purpose |
|---------|---------|
| `cv2` (OpenCV) | Grayscale conversion (BGR2GRAY), image analysis |
| `numpy` | Array operations, mask creation, pixel statistics |
| `yaml` | Load `dark_bg.*` configuration from `config.yaml` |
| `inferencer.face_pipeline` | `quality_detector` (Stage 1 ONNX model), `face_detector` (Stage 2 face bbox), `StatusCode.NOFACE` |

### Design Patterns Applied

- **Pipeline Pattern**: Sequential two-stage processing with early exits at each stage
- **Configuration Externalization**: All thresholds loaded from `config.yaml` (`dark_bg.*` namespace)

### Data Flow

```
Input (np.ndarray BGR image)
    │
    ├─ Stage 1: quality_detector.detect(img) → 4-class scores
    │     ├─ argmax ≠ 2 or dark_score < 0.95 → return (False, 100, dark_score, 0.0, 0.0, 0.0, 0.0)
    │     └─ pass → proceed to Stage 2
    │
    └─ Stage 2: face_detector.get_face_bboxes(img)
          ├─ no faces → return (False, 300, dark_score, 0.0, 0.0, 0.0, 0.0)
          └─ has faces → detect_bg_darkness(img, face_bbox)
                ├─ bg_pixels == 0 → return (False, -1.0, 0.0, 0.0, 0.0) [sentinel: no bg → fallback to Stage1]
                ├─ bg_pixels > 0:
                │     dark_ratio = (bg_pixels < dark_pixel_thresh).mean()
                │     spot_ratio, patch_ratio = classify_bright_regions(gray, mask, bright_thresh, min_patch_area)
                │     is_dark = dark_ratio > dark_ratio_thresh AND patch_ratio < patch_ratio_thresh AND spot_ratio < spot_ratio_thresh
                │     return (bool, dark_ratio, spot_ratio, patch_ratio, darkness_level)

Output: (is_dark_bg: bool, resp_code: int, dark_score: float, dk_ratio: float, spot_ratio: float, patch_ratio: float, darkness_level: float)
```

---

## 3. Core Components Deep Dive

### `classify_bright_regions(gray, mask, bright_thresh, min_patch_area)`

| Item | Detail |
|------|--------|
| **Purpose** | Segment background bright regions into small specular spots and large continuous patches using connected components analysis |
| **Parameters** | `gray`: grayscale image; `mask`: background mask (255=bg, 0=face/below); `bright_thresh`: pixels > this are treated as "bright"; `min_patch_area`: area threshold splitting spot vs patch |
| **Return** | `(spot_ratio, patch_ratio)` — proportion of background pixels occupied by small spots and large patches respectively |
| **Core Logic** | 1. Zero out non-background pixels; 2. Binary threshold at `bright_thresh`; 3. Morphological opening (3×3 ellipse) for denoising; 4. `cv2.connectedComponentsWithStats` with 8-connectivity; 5. Sum area per label: area < `min_patch_area` → spot, else → patch; 6. Normalize by total background pixel count |

---

### `detect_bg_darkness(img_bgr, face_bbox)`

| Item | Detail |
|------|--------|
| **Purpose** | CV-based dark background analysis with connected components for bright region classification |
| **Parameters** | `img_bgr`: `np.ndarray` (BGR, HWC, uint8); `face_bbox`: `[x1, y1, x2, y2]` (int) |
| **Return** | `(bool, float, float, float, float)` — `is_dark`, `dark_ratio`, `spot_ratio`, `patch_ratio`, `darkness_level`. Returns `(False, -1.0, 0.0, 0.0, 0.0)` as sentinel when no background pixels exist. |
| **Core Logic** | 1. Convert to grayscale; 2. Expand bbox (up: 10%, other: 0%); 3. Clip to image boundaries; 4. Create mask: face region = 0, area below face = 0, bg = 255; 5. Compute `dark_ratio` = proportion of bg pixels < `dark_pixel_thresh`; 6. Call `classify_bright_regions` to get `spot_ratio`, `patch_ratio`; 7. `is_dark = dark_ratio > dark_ratio_thresh AND patch_ratio < patch_ratio_thresh AND spot_ratio < spot_ratio_thresh`; 8. If `is_dark`, compute `darkness_level` from mean brightness of non-dark bg pixels |
| **Design Rationale** | The three-condition judgment (dark pixels, small spots, large patches) was adopted to distinguish: **night car interior with specular reflections** (acceptable: high dark_ratio, low spot_ratio, low patch_ratio) from **daytime car interior with bright glass/seats** (rejected: patch_ratio elevated). The old two-threshold approach (dark_ratio + bright_ratio) could not separate specular reflections from large gray/bright areas, causing false positives. |

---

### `dark_bg_check(img_bgr)`

| Item | Detail |
|------|--------|
| **Purpose** | Unified two-stage dark background detection entry point |
| **Parameters** | `img_bgr`: `np.ndarray` (BGR) — full input image |
| **Return** | `(is_dark_bg, resp_code, dark_score, dk_ratio, spot_ratio, patch_ratio, darkness_level)` — 7-tuple: final judgment with debug scores. Early exits (Stage 1 fail, no face, sentinel) also return 7 values with zeros for uncomputed fields |
| **Core Logic** | Stage 1: model inference → early exit if not dark class or score < threshold; Stage 2: detect face bbox → no face → resp=300; has face → CV analysis; if dk_ratio == -1.0 (sentinel, no bg pixels) → fallback to stage1 result; otherwise → stage2 result with spot_ratio/patch_ratio |

---

## 4. Constraints, Edge Cases & Side Effects

### Preconditions

- Model files loaded by `face_pipeline` (`weights-finetune-1-40--A.onnx`, `arcface_weights_best_new.onnx`, `shape_predictor_68_face_landmarks.dat`) must exist at `./resources/`
- `config.yaml` must contain `dark_bg` section with all required keys (falls back to hardcoded defaults if missing or partial)
- Input image must be BGR `np.ndarray` (uint8, 3 channels)

### Known Limitations

| Limitation | Detail |
|------------|--------|
| numpy `bool_` | `(bg_pixels < thresh).mean()` returns a `numpy.float64`, and the comparison `> _dark_ratio_thresh` yields a `numpy.bool_` which is not JSON-serializable. Explicitly cast with `bool()` before returning to the API layer (`server.py` uses `jsonify`). |
| `min_patch_area` is resolution-dependent | Currently set to `500` px for 640×480 input; different resolutions may require tuning. A 2000×2000 image would need a proportionally larger threshold to avoid classifying moderate-sized features as "patches". |
| Stage 1 model specificity | `weights-finetune-1-40--A.onnx` is an internal model; availability tied to this project |
| Single-face assumption | Stage 2 uses only the first detected face bbox; multiple faces are ignored |

### Side Effect Warnings

- `face_pipeline` module import loads ONNX models into GPU memory (persists for process lifetime)
- Module reads `config.yaml` at import time (file system I/O)
- No writes to file system during inference

---

## 5. Vibe Coding Extension & Maintenance Guide

### Configurable Parameters (`config.yaml` → `dark_bg`)

| Parameter | Current | Description |
|-----------|---------|-------------|
| `model_threshold` | `0.95` | Stage 1: minimum class-2 score for dark category |
| `dark_pixel_thresh` | `30` | Pixels < this are counted as "dark" for `dark_ratio` |
| `dark_ratio_thresh` | `0.90` | Minimum proportion of dark background pixels required |
| `bright_thresh` | `60` | Pixels > this enter connected components analysis |
| `min_patch_area` | `500` | Connected region area ≥ this → large patch (glass/seat); < this → specular spot |
| `patch_ratio_thresh` | `0.02` | Maximum allowed proportion of large bright patches |
| `spot_ratio_thresh` | `0.10` | Maximum allowed proportion of small specular spots |
| `bbox_expand_ratio` | `0.0` | Horizontal expansion ratio for face bbox |
| `bbox_expand_up_ratio` | `0.1` | Upward expansion ratio for face bbox (excludes clothing) |

### Safe Modification Zones

| Zone | Safety | Notes |
|------|--------|-------|
| Config thresholds in `config.yaml` | Safe | All `dark_bg.*` values can be tuned without code changes |
| `detect_bg_darkness` expand ratios | Safe | Can adjust up/down expand ratios independently (defaults: up=10%, other=0%) |
| `bright_thresh` | Safe | Lower → more pixels enter connected components analysis (stricter); Higher → fewer (looser) |
| `min_patch_area` | Safe | Lower → more regions classified as "patch" (stricter); Higher → more regions classified as "spot" (looser) |
| `patch_ratio_thresh` / `spot_ratio_thresh` | Safe | Directly control acceptance rate for large bright regions vs specular spots |
| Stage 1 model path | Moderate | Must match model input/output spec (320×240, 4-class) |
| `dark_bg_check` return tuple format | Caution | `server.py` expects `(bool, int, float, float, float, float, float)` — 7-tuple |

### Common Refactoring Pitfalls

- **Adding `/255` normalization to Stage 1**: The model expects raw `[0, 255]` float32 input; normalizing would break inference
- **Changing `StatusCode.NOFACE` import**: If the import is removed and 300 is hardcoded, it becomes decoupled from the project's status code convention

### Testing Recommendations

- **Self-test via `__main__`**: Run `python inferencer/dark_bg_detector.py` to test with `./unit_test/test_img/black_bg_test.jpg` directly (requires model files in `./resources/`).
- **Night car interior (dark background)**: Verify `is_dark_bg=True`, `dark_score ≥ 0.95`, `dk_ratio > 0.90`, `patch_ratio < 0.02`, `spot_ratio < 0.10`, `darkness_level > 70`
- **Night car + specular reflections**: Verify `is_dark_bg=True` — spots on glass/plastic pass (low `patch_ratio`, moderate `spot_ratio`)
- **Daytime car with bright glass**: Verify `is_dark_bg=False` — glass region triggers elevated `patch_ratio`
- **Daytime car with gray seats**: Verify `is_dark_bg=False` — seat region triggers elevated `patch_ratio`
- **Normal lighting image**: Verify `is_dark_bg=False` (should exit at Stage 1 or Stage 2 with low `dark_ratio`)
- **No face image**: Verify `resp_code=300`, `is_dark_bg=False`
- **Face-filling image (no background)**: Verify fallback to Stage 1 result when `dk_ratio=-1.0` (sentinel)
- **Config hot-reload test**: Change `dark_bg.patch_ratio_thresh` in yaml and verify new threshold takes effect on restart
