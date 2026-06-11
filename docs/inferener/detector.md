# detector.py - Module Documentation

## 1. Module Overview

**Core Responsibility:** Orchestrates the full face detection pipeline — from YOLO-based bounding box detection to dlib-based face alignment — producing aligned face chips ready for downstream recognition models.

**Key Functionalities:**
- Detect faces and facial components (eyes, mouth) via YOLO model
- Validate face completeness (ensures eyes and mouth are present within face bounding box)
- Perform 68-landmark-based face alignment using dlib
- Return aligned face chips and per-face confidence scores

---

## 2. Architecture & Design Patterns

### Dependencies

| Library | Purpose |
|---------|---------|
| `dlib` | 68-point facial landmark prediction and face alignment (`get_face_chip`) |
| `cv2` (OpenCV) | Color space conversion (BGR↔RGB) and image I/O |
| `collections.defaultdict` | Group detections by class label |
| `inferencer.geo_check.complete_face` | Validate facial feature completeness |
| `inferencer.yolo_detection.YoloDetection` | YOLO-based object detection |

### Design Patterns Applied

- **Facade Pattern**: `Detector` provides a single high-level `get_face_capture` method that hides the complexity of detection → validation → alignment.
- **Pipeline Pattern**: Sequential processing stages with early-exit on failure (no detections / no faces).

### Data Flow

```
Input (np.ndarray BGR image)
    -> YoloDetection.run_detect: produce bounding boxes [x1,y1,x2,y2,conf,cls]
    -> Filter: early return if no detections or no face class
    -> Group by class: {0: faces, 1: left_eye, 2: right_eye, 3: mouth}
    -> BGR→RGB conversion (once)
    -> complete_face: validate each face has eyes + mouth within its bbox
    -> dlib_wrap: landmark detection + face alignment per valid face
    -> Output: (face_chip_list: List[np.ndarray], face_conf: List[int]) (aligned, same length)
```

---

## 3. Core Components Deep Dive

### `Detector.__init__(self, weights, sp)`

| Item | Detail |
|------|--------|
| **Purpose** | Initialize YOLO detector and dlib landmark predictor |
| **Parameters** | `weights`: `str` - path to YOLO ONNX model; `sp`: `str` - path to dlib 68-landmark `.dat` file |
| **Return** | None |
| **Core Logic** | Loads `dlib.shape_predictor` for landmark detection; instantiates `YoloDetection` for face/component bounding box prediction |

---

### `Detector.dlib_wrap(self, img_rgb, x1, y1, x2, y2, size)`

| Item | Detail |
|------|--------|
| **Purpose** | Align a face region using dlib's 68-landmark model |
| **Parameters** | `img_rgb`: RGB image (pre-converted by caller); `x1, y1, x2, y2`: face bounding box coordinates (int); `size`: output chip size in pixels |
| **Return** | `np.ndarray` - aligned face image (BGR, `size x size`) |
| **Core Logic** | Predicts 68 landmarks within the given rectangle, calls `dlib.get_face_chip` to produce an affine-aligned face, converts RGB→BGR for output. Color conversion is done once by the caller for efficiency |

---

### `Detector.get_face_capture(self, pic, size=112)`

| Item | Detail |
|------|--------|
| **Purpose** | End-to-end face detection, validation, and alignment |
| **Parameters** | `pic`: `np.ndarray` (BGR); `size`: `int` - aligned face output size (default 112 for ArcFace) |
| **Return** | `(face_chip_list, face_conf)` — list of aligned face images + list of confidence percentages (0-100), guaranteed to be the same length; returns `([], [])` on failure |
| **Core Logic** | Runs YOLO detection → early exits if no faces → groups detections by class into `face_dict` → converts image to RGB once → validates completeness via `complete_face` → aligns each valid face via `dlib_wrap` and collects corresponding confidence |

---

## 4. Constraints, Edge Cases & Side Effects

### Preconditions

- YOLO ONNX model and dlib `.dat` landmark file must exist at specified paths
- Input image must be BGR `np.ndarray` (uint8, 3 channels)
- `complete_face` expects `face_dict` with int keys `{0: faces, 1: left_eye, 2: right_eye, 3: mouth}`

### Known Limitations

- **Single-scale detection**: No image pyramid; small faces in high-res images may be missed
- **No batch processing**: Processes one image at a time
- **dlib alignment assumes frontal/near-frontal faces**: Extreme pose angles produce poor alignment

### Side Effect Warnings

- `sys.path.append` at module level permanently mutates `sys.path` for the process
- dlib landmark prediction allocates internal buffers; no external I/O or global state changes
- Input image `pic` is NOT modified (read-only access)

---

## 5. Vibe Coding Extension & Maintenance Guide

### Safe Modification Zones

| Zone | Safety | Notes |
|------|--------|-------|
| `__main__` block | Safe | Test code only |
| `size` default value (112) | Safe | Can change to match different recognition models (e.g., 160 for FaceNet) |
| `face_conf` formatting | Safe | `int(i * 100)` can be changed to float or different scale |
| `dlib_wrap` internals | Moderate | Alignment logic can be swapped (e.g., to OpenCV affine) without API change |
| `get_face_capture` return format | Caution | Downstream consumers depend on `(list, list)` tuple |

### Common Refactoring Pitfalls

- **Changing `face_dict` key type**: Must stay consistent with `geo_check.complete_face` which expects int keys `0, 1, 2, 3`
- **Removing `complete_face` validation**: Will allow incomplete/occluded faces through, degrading recognition accuracy
- **Passing BGR to `dlib_wrap`**: The method now expects pre-converted RGB input; passing BGR will produce wrong landmarks and misaligned output
- **Confidence indexing**: `face_conf` uses `faces[ii]` to index; if detection order changes or filtering is modified, ensure index alignment is maintained

### Testing Recommendations

- **Integration test**: Provide an image with known face count; verify `len(face_chip_list)` matches expected
- **Incomplete face test**: Use an image with occluded face (missing mouth); verify it is excluded from results
- **Alignment quality test**: Compare aligned face output against a reference alignment to verify landmark accuracy
- **Empty input test**: Verify `([], [])` is returned for images with no faces or blank images
