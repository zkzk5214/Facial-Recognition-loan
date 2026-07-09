# face_detector.py - Module Documentation

## 1. Module Overview

**Core Responsibility:** Provides the complete face detection pipeline — from YOLO-based bounding box detection, facial component validation (eyes + mouth), rotation correction, to dlib-based 68-landmark face alignment — producing aligned face chips ready for downstream recognition.

**Key Functionalities:**
- Detect faces and facial components (eyes, mouth) via YOLO model
- Validate face completeness (eyes and mouth present, non-overlapping, within face bbox)
- Compute face rotation angle based on eye alignment and apply rotation correction
- Retry incomplete faces by expanding the bounding box and re-detecting
- Perform 68-landmark-based face alignment using dlib

---

## 2. Architecture & Design Patterns

### Dependencies

| Library | Purpose |
|---------|---------|
| `dlib` | 68-point facial landmark prediction and face alignment (`get_face_chip`) |
| `cv2` (OpenCV) | Color conversion, image rotation (`warpAffine`), resize |
| `numpy` | Vector operations, angle calculations, trigonometry |
| `collections.defaultdict` | Group detections by class label |
| `inferencer.yolo_detection.YoloDetection` | YOLO-based object detection |

### Design Patterns Applied

- **Facade Pattern**: `Detector.get_face_capture` provides a single high-level API hiding detection → validation → rotation → alignment complexity.
- **Pipeline Pattern**: Sequential processing stages with early-exit on failure.
- **Template Method**: `_rotate_and_align` conditionally applies rotation before the common alignment step.
- **Retry Pattern**: `_retry_incomplete_face` expands bbox and re-detects when initial validation fails.

### Data Flow

```
Input (np.ndarray BGR image)
    -> YOLO detection: [x1,y1,x2,y2,conf,cls] for faces/eyes/mouth
    -> _build_face_dict: group by class {0:faces, 1:left_eye, 2:right_eye, 3:mouth}
    -> BGR → RGB (once)
    -> For each face:
        -> complete_face: validate components + compute rotation params
        -> Complete? → _rotate_and_align → aligned face (RGB)
        -> Incomplete? → _retry_incomplete_face → expand bbox → re-detect → re-validate → align
    -> RGB → BGR (once per face chip)
    -> Output: (face_chip_list, face_conf)
```

---

## 3. Core Components Deep Dive

### Utility Functions

| Function | Purpose |
|----------|---------|
| `centre_bbx(bbox)` | Return center point (cx, cy) of a bbox |
| `check_in_face(bbox_list, face_bbox)` | Filter components whose center is inside the face bbox |
| `cal_iou(box1, box2)` | Standard IoU calculation |
| `del_dup(bbox_list, iou_thres)` | Greedy IoU-based duplicate removal |
| `check_intersection(box1, box2)` | Check if two boxes have any overlap |
| `vector_angle(v1, v2)` | Full 0°-360° angle between two 2D vectors using 2D cross product |
| `get_rotation_matrix(p1, p2, ...)` | Compute affine rotation matrix with canvas expansion |
| `calculate_eye_rotation(face, right, left)` | Compute rotation based on eye positions relative to face bbox |
| `cal_angle(a, b, c)` | Signed angle at point A between vectors AB and AC |
| `complete_face(face_dict)` | Validate completeness + compute rotation; returns `[[bool, [M, angle] or []], ...]` |
| `_build_face_dict(res)` | Group detection results into dict by class label |
| `_get_bbox(face_dict, idx)` | Extract integer bbox from face_dict |

---

### `Detector._align_face(self, img_rgb, x1, y1, x2, y2, size)`

| Item | Detail |
|------|--------|
| **Purpose** | Align a face using dlib's 68-landmark model |
| **Parameters** | `img_rgb`: RGB image; `x1,y1,x2,y2`: face bbox; `size`: output chip size |
| **Return** | `np.ndarray` — aligned face (RGB, `size x size`) |
| **Core Logic** | Predict 68 landmarks within rectangle, ensure contiguous memory via `np.ascontiguousarray`, call `dlib.get_face_chip` for affine alignment |
| **Contiguity Fix** | `np.ascontiguousarray(img_rgb)` prevents pybind11 type mismatch when non-contiguous numpy views (e.g., crop slices) are passed to dlib |

---

### `Detector._rotate_and_align(self, img_rgb, rotate_para, x1, y1, x2, y2, size, angle_threshold)`

| Item | Detail |
|------|--------|
| **Purpose** | Conditionally rotate a tilted face, then align with dlib |
| **Parameters** | `rotate_para`: `[rotation_matrix, angle]` or `[]`; `angle_threshold`: rotation trigger threshold (default 30°) |
| **Return** | `np.ndarray` — aligned face (RGB) |
| **Core Logic** | If rotation angle > threshold: crop face → compute new canvas size → `warpAffine` → align. Otherwise: align directly |

---

### `Detector._retry_incomplete_face(self, pic_rgb, x1, y1, x2, y2, size, angle_threshold)`

| Item | Detail |
|------|--------|
| **Purpose** | Recover incomplete faces by expanding bbox 15% and re-detecting |
| **Parameters** | Same as face bbox coordinates + alignment params |
| **Return** | Aligned face (RGB) or `None` if retry fails |
| **Core Logic** | Expand bbox by `EXPAND_RATIO` → crop with `.copy()` for contiguous array → convert to BGR for YOLO → re-detect → validate exactly 1 complete face → rotate and align |
| **Contiguity Fix** | `.copy()` on the crop slice ensures the array is C-contiguous before passing through `_rotate_and_align` to dlib |

---

### `Detector.get_face_capture(self, pic, angle_threshold=30, size=112)`

| Item | Detail |
|------|--------|
| **Purpose** | End-to-end face detection, validation, and alignment |
| **Parameters** | `pic`: BGR image; `angle_threshold`: rotation correction threshold; `size`: output face chip size |
| **Return** | `(face_chip_list, face_conf)` — aligned BGR face images + confidence percentages, guaranteed same length |
| **Core Logic** | YOLO detect → build face dict → BGR→RGB once → for each face: validate → complete? rotate+align : retry → collect results with RGB→BGR conversion |

---

## 4. Constraints, Edge Cases & Side Effects

### Preconditions

- YOLO ONNX model and dlib 68-landmark `.dat` file must exist
- Input image must be BGR `np.ndarray` (uint8, 3 channels)
- `complete_face` expects `face_dict` with string keys `'0','1','2','3'`

### Known Issues & Resolved Bugs

- **2026-07-09: dlib TypeError on non-contiguous arrays** — `_retry_incomplete_face`中切片`pic_rgb[...:... , :]`产生numpy view，非C-contiguous。当走非旋转分支时，该view直传`dlib.get_face_chip()`，pybind11参数匹配失败抛`TypeError`。修复：切片处加`.copy()`，`_align_face`中加`np.ascontiguousarray()`兜底，保证传给dlib的数组始终连续。

### Known Limitations

- **Retry adds latency**: Each incomplete face triggers a second YOLO inference on the expanded crop
- **Single retry only**: If the expanded bbox still doesn't produce a complete face, it's skipped (no further retries)
- **Rotation threshold is coarse**: 30° default may miss moderately tilted faces (15-30°) that dlib handles poorly
- **`vector_angle` numerical edge case**: `np.arccos` may produce NaN if dot product exceeds [-1, 1] due to floating point
- **Face confidence alignment assumption**: `faces[idx]` assumes detection order matches `face_dict['0']` order

### Side Effect Warnings

- BGR→RGB conversion creates a copy of the full image (memory)
- `_retry_incomplete_face` creates an additional BGR copy for YOLO re-detection
- No file I/O or global state mutations

---

## 5. Vibe Coding Extension & Maintenance Guide

### Safe Modification Zones

| Zone | Safety | Notes |
|------|--------|-------|
| `EXPAND_RATIO` (0.15) | Safe | Tunable; larger = more aggressive retry |
| `angle_threshold` default (30) | Safe | Lower = more faces get rotated; higher = more rely on dlib |
| `_get_bbox` / `_build_face_dict` | Safe | Pure helpers |
| Rotation logic in `complete_face` | Moderate | Affects which faces get rotation correction |
| `get_face_capture` return format | Caution | `face_pipeline.py` depends on `(list, list)` |

### Common Refactoring Pitfalls

- **Changing color space flow**: The pipeline is RGB internally; breaking this assumption causes silent color errors in dlib alignment
- **Modifying `face_dict` key type**: Must stay as strings; `complete_face` reads `'0','1','2','3'`
- **Removing retry logic**: Some edge-positioned faces will be lost; affects recall
- **Changing `_align_face` to accept BGR**: Would require updating `_rotate_and_align` and `_retry_incomplete_face` call sites

### Testing Recommendations

- **Complete face test**: Image with clear frontal face → verify chip returned with high confidence
- **Tilted face test**: Image with >30° head tilt → verify rotation correction produces horizontal eyes
- **Incomplete face test**: Partially occluded face → verify retry recovers it (or gracefully skips)
- **Multi-face test**: Image with multiple faces → verify correct count and independent confidence values
- **Edge face test**: Face at image boundary → verify bbox expansion is clamped correctly
