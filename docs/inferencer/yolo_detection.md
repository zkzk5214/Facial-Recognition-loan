# yolo_detection.py - Module Documentation

## 1. Module Overview

**Core Responsibility:** Provides YOLO-based object detection (face, eyes, mouth) using an ONNX model with letterbox preprocessing and NMS post-processing.

**Key Functionalities:**
- Load and initialize a YOLO ONNX detection model with CUDA/CPU execution support
- Perform letterbox image preprocessing (resize + pad) to meet model input requirements
- Run inference and apply Non-Maximum Suppression to produce final bounding box detections
- Rescale detection coordinates back to original image space

---

## 2. Architecture & Design Patterns

### Dependencies

| Library | Purpose |
|---------|---------|
| `cv2` (OpenCV) | Image resize, padding, and I/O |
| `torch` | Tensor operations for NMS and result aggregation |
| `onnxruntime` | ONNX model inference engine |
| `numpy` | Array manipulation and type conversion |
| `torchvision` | `torchvision.ops.nms` for GPU-accelerated NMS |

### Design Patterns Applied

- **Encapsulation Pattern**: `YoloDetection` class encapsulates model loading, preprocessing, inference, and post-processing into a unified interface with a single `run_detect` entry point.
- **Pipeline Pattern**: Data flows through a sequential pipeline — letterbox → normalize → inference → NMS → coordinate rescaling.

### Data Flow

```
Input (np.ndarray BGR image, HWC)
    -> letterbox: resize + pad to (640, 640), transpose to CHW, BGR->RGB
    -> Normalize: add batch dim, cast float32, divide by 255
    -> ONNX inference: predict raw detections [b, 25500, 9]
    -> NMS: filter by confidence + IoU thresholds
    -> scale_coords: map boxes back to original image dimensions
    -> Output: np.ndarray of shape (N, 6) [x1, y1, x2, y2, conf, cls]
```

---

## 3. Core Components Deep Dive

### `YoloDetection.__init__(self, weights_add)`

| Item | Detail |
|------|--------|
| **Purpose** | Initialize model session, cache I/O metadata, and configure target image size |
| **Parameters** | `weights_add`: `str` - path to ONNX model file |
| **Return** | None |
| **Core Logic** | Creates an ONNX InferenceSession with CUDA preferred (CPU fallback); caches `self.input_name` and `self.output_name` for inference calls; sets fixed input size `(640, 640)` |

---

### `YoloDetection.letterbox(self, img, color=(114, 114, 114))`

| Item | Detail |
|------|--------|
| **Purpose** | Resize and pad an image to the model's expected input size while preserving aspect ratio |
| **Parameters** | `img`: `np.ndarray` (HWC, BGR); `color`: `tuple` - border fill color |
| **Return** | `np.ndarray` (CHW, RGB, uint8) - preprocessed image |
| **Core Logic** | Computes scale ratio to fit image within `self.imgsz`, resizes with bilinear interpolation, applies symmetric padding, transposes HWC→CHW and reverses BGR→RGB |

---

### `YoloDetection.run_detect(self, input_image, conf_thres, iou_thres, max_det, agnostic_nms)`

| Item | Detail |
|------|--------|
| **Purpose** | Execute full detection pipeline on a single image |
| **Parameters** | `input_image`: `np.ndarray` (HWC, BGR); `conf_thres`: `float` - confidence threshold (default 0.3); `iou_thres`: `float` - IoU threshold for NMS (default 0.45); `max_det`: `int` - max detections (default 1000); `agnostic_nms`: `bool` - class-agnostic NMS flag |
| **Return** | `np.ndarray` of shape `(N, 6)` — each row is `[x1, y1, x2, y2, confidence, class_id]`; returns empty `(0, 6)` array if no detections |
| **Core Logic** | Preprocesses via letterbox, normalizes to `[0, 1]` float32, runs ONNX session, applies NMS, rescales coordinates to original image dimensions, and aggregates all valid detections |

---

### Post-processing Functions (module-level)

| Function | Purpose |
|----------|---------|
| `clip_coords(boxes, shape)` | Clamp xyxy bounding boxes to image boundaries (height, width) using `torch.clamp_` |
| `scale_coords(img1_shape, coords, img0_shape)` | Rescale xyxy coords from letterboxed image space back to original image; removes padding then divides by gain |
| `xywh2xyxy(x)` | Convert boxes from `[cx, cy, w, h]` to `[x1, y1, x2, y2]` format (returns a clone) |

---

### `non_max_suppression(prediction, conf_thres, iou_thres, max_det, agnostic)`

| Item | Detail |
|------|--------|
| **Purpose** | Perform Non-Maximum Suppression on raw YOLO model output |
| **Parameters** | `prediction`: `Tensor` (batch, num_anchors, 5+num_classes); `conf_thres`: objectness threshold; `iou_thres`: NMS IoU threshold; `max_det`: max detections per image; `agnostic`: class-agnostic NMS flag |
| **Return** | `List[Tensor]` — one tensor per batch image, each shape `(n, 6)` as `[x1, y1, x2, y2, conf, cls]` |
| **Core Logic** | For each image: filter by confidence → zero out invalid-sized boxes → compute `obj_conf * cls_conf` → convert xywh→xyxy → select best class per box → apply `torchvision.ops.nms` with per-class offset → limit to `max_det` |

---

## 4. Constraints, Edge Cases & Side Effects

### Preconditions

- ONNX model file must exist at the specified path and be compatible with input shape `(1, 3, 640, 640)`
- Input image must be a valid BGR `np.ndarray` with 3 channels (uint8)
- Post-processing functions (`non_max_suppression`, `scale_coords`) are defined within this module
- For GPU acceleration: CUDA and cuDNN properly installed

### Known Limitations

- **Single-image inference only**: No native batch support; processes one image per call
- **Fixed input size**: Hardcoded to `640x640`; cannot be adjusted without modifying `self.imgsz`
- **Torch dependency for post-processing**: Uses PyTorch tensors for NMS even though inference is via ONNX Runtime, adding an unnecessary heavy dependency
- **Output column count assumption**: Returns `np.empty((0, 6))` for empty results — assumes 6 columns (xyxy + conf + cls); if model outputs differ, this may cause shape mismatches downstream

### Side Effect Warnings

- Model loading allocates GPU memory (if CUDA provider available) that persists for the object's lifetime
- `non_max_suppression` modifies the input `prediction` tensor in-place (zeroing objectness for invalid boxes)
- No file system writes or network requests during inference

---

## 5. Vibe Coding Extension & Maintenance Guide

### Safe Modification Zones

| Zone | Safety | Notes |
|------|--------|-------|
| `__main__` block | Safe | Test/demo code; no external dependents |
| `conf_thres` / `iou_thres` defaults | Safe | Tunable hyperparameters |
| `letterbox` color default | Safe | Cosmetic; `(114, 114, 114)` is conventional gray |
| `self.imgsz` value | Caution | Must match model's trained input resolution |
| `run_detect` return format | Caution | Downstream consumers depend on `(N, 6)` shape |

### Common Refactoring Pitfalls

- **Changing `self.imgsz`**: The ONNX model has a fixed input shape; mismatched sizes will cause runtime errors or incorrect outputs
- **Removing `scale_coords`**: Detections will remain in `640x640` space instead of original image coordinates
- **Modifying letterbox padding logic**: The `±0.1` rounding trick ensures pixel-exact symmetry; naive rounding breaks coordinate mapping
- **Replacing torch with numpy for NMS**: `non_max_suppression` expects torch tensors; switching requires rewriting the entire post-processing pipeline

### Testing Recommendations

- **Functional test**: Run detection on a known image with labeled ground-truth boxes; verify IoU > 0.5 with expected results
- **Empty detection test**: Feed a blank/uniform image and verify the `(0, 6)` empty array is returned cleanly
- **Aspect ratio test**: Test with extreme aspect ratios (very wide / very tall) to confirm letterbox padding is correct
- **Threshold boundary test**: Verify that lowering `conf_thres` increases detection count and raising it decreases count
