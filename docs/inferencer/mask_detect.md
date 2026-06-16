# mask_detect.py - Module Documentation

## 1. Module Overview

**Core Responsibility:** Provides mask/no-mask classification for face images using an ONNX classification model, determining whether a person in the image is wearing a face mask.

**Key Functionalities:**
- Load an ONNX mask detection model at module level with CUDA/CPU support
- Preprocess input images (resize, normalize) for model inference
- Classify images and return the predicted class index

---

## 2. Architecture & Design Patterns

### Dependencies

| Library | Purpose |
|---------|---------|
| `cv2` (OpenCV) | Image resize |
| `onnxruntime` | ONNX model inference engine |
| `numpy` | Array manipulation, normalization, argmax |

### Design Patterns Applied

- **Module-level Singleton**: Model is loaded once at import time and reused across all `get_class` calls. No class encapsulation — designed for simple stateless function calls.

### Data Flow

```
Input (np.ndarray BGR image, HWC, uint8)
    -> cv2.resize to (224, 224)
    -> Add batch dim + cast float32 + normalize to [0, 1]
    -> ONNX inference
    -> argmax on output logits
    -> Output: int (class index, e.g., 0=mask, 1=no_mask)
```

---

## 3. Core Components Deep Dive

### Module-level Initialization

| Item | Detail |
|------|--------|
| `model` | ONNX InferenceSession loaded from `./resources/maskModel.onnx` |
| `_input_name` | Cached model input tensor name for efficient repeated inference |

---

### `get_class(img)`

| Item | Detail |
|------|--------|
| **Purpose** | Classify whether the input image contains a masked or unmasked face |
| **Parameters** | `img`: `np.ndarray` — input image (HWC, BGR, uint8, any size) |
| **Return** | `int` — predicted class index (0 or 1) |
| **Core Logic** | Resizes to 224x224, normalizes to [0,1] float32, runs ONNX inference, returns argmax of output logits |

---

## 4. Constraints, Edge Cases & Side Effects

### Preconditions

- ONNX model file must exist at `./resources/maskModel.onnx`
- Input image must be a valid BGR `np.ndarray` with 3 channels (uint8)
- Working directory must be the project root (due to hardcoded relative model path)

### Known Limitations

- **Hardcoded model path**: Cannot configure model location without modifying source code
- **No class encapsulation**: Unlike other inferencer modules, uses module-level globals; harder to test in isolation
- **No batch support**: Single-image inference only
- **Class semantics undocumented**: The meaning of output class indices (0 vs 1) depends on model training; not explicitly defined in code
- **No confidence score returned**: Only returns argmax class, not the probability

### Side Effect Warnings

- Module import triggers model loading and GPU memory allocation
- No file I/O or global state mutations during inference

---

## 5. Vibe Coding Extension & Maintenance Guide

### Safe Modification Zones

| Zone | Safety | Notes |
|------|--------|-------|
| `__main__` block | Safe | Test code only |
| Resize dimensions `(224, 224)` | Caution | Must match model's trained input resolution |
| Normalization (`/ 255.0`) | Caution | Must match model's training preprocessing |
| Model path | Moderate | Can change if model file is relocated |

### Common Refactoring Pitfalls

- **Adding /255 differently**: Model expects [0, 1] input; changing normalization breaks predictions
- **Changing resize dimensions**: Must match the model's expected spatial input
- **Converting to class-based**: Would require updating `server.py` import (`from inferencer import mask_detect` → instance-based call)
- **Interpreting class index**: Ensure the caller (`server.py`) correctly maps class index to mask/no-mask semantics

### Testing Recommendations

- **Mask test**: Provide image with clearly visible mask; verify returns the "mask" class index
- **No-mask test**: Provide image without mask; verify returns the "no mask" class index
- **Shape test**: Verify model handles various input aspect ratios after resize
