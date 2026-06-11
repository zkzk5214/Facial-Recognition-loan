# image_quality.py - Module Documentation

## 1. Module Overview

**Core Responsibility:** Provides image quality classification using an ONNX model to assess whether an input image meets quality standards (e.g., lighting, exposure, noise level).

**Key Functionalities:**
- Load and initialize an ONNX image quality classification model with CUDA/CPU support
- Preprocess input images (resize to fixed dimensions) for model inference
- Run inference and return raw class probability scores for quality assessment

---

## 2. Architecture & Design Patterns

### Dependencies

| Library | Purpose |
|---------|---------|
| `onnxruntime` | ONNX model inference engine |
| `numpy` | Array manipulation and type conversion |
| `cv2` (OpenCV) | Image resize and I/O |

### Design Patterns Applied

- **Encapsulation Pattern**: `ImgQuality` class wraps model loading, input name caching, preprocessing, and inference into a minimal interface with a single `detect` method.

### Data Flow

```
Input (np.ndarray BGR image, HWC, uint8)
    -> cv2.resize to (width=240, height=320)
    -> Add batch dim + cast to float32 (no normalization, [0-255] range)
    -> ONNX inference
    -> Output: np.ndarray of class scores (raw logits/probabilities)
```

---

## 3. Core Components Deep Dive

### `ImgQuality.__init__(self, weights)`

| Item | Detail |
|------|--------|
| **Purpose** | Initialize ONNX model session and cache input metadata |
| **Parameters** | `weights`: `str` - path to ONNX model file |
| **Return** | None |
| **Core Logic** | Creates an ONNX InferenceSession with CUDA preferred (CPU fallback); caches `self.input_name` for efficient repeated inference |

---

### `ImgQuality.detect(self, img)`

| Item | Detail |
|------|--------|
| **Purpose** | Run quality classification on a single image |
| **Parameters** | `img`: `np.ndarray` - input image (HWC, BGR, uint8) |
| **Return** | `np.ndarray` - 1D array of class scores; higher score at index indicates stronger class membership |
| **Core Logic** | Resizes image to `(240, 320)` (width, height), adds batch dimension, casts to float32 (no /255 normalization), runs ONNX inference, returns first sample's output |

---

### `detect_quality` (in `__main__`)

| Item | Detail |
|------|--------|
| **Purpose** | Wrapper that interprets model output as a binary quality decision |
| **Parameters** | `img`: input image; `return_value`: `bool` - whether to also return raw scores |
| **Return** | `bool` (True if class 3 = good quality) or `(bool, np.ndarray)` if `return_value=True` |
| **Core Logic** | Takes argmax of model output; checks if top class is index 3 (assumed "good quality" class) |

---

## 4. Constraints, Edge Cases & Side Effects

### Preconditions

- ONNX model file must exist at specified path and expect input shape `(1, 320, 240, 3)` with float32 in `[0, 255]` range
- Input image must be a valid BGR `np.ndarray` with 3 channels (uint8)
- For GPU acceleration: CUDA and cuDNN properly installed

### Known Limitations

- **Fixed resize without aspect ratio preservation**: All images are resized to `(240, 320)` regardless of original aspect ratio, introducing potential distortion
- **No input normalization**: Model expects raw `[0, 255]` float values; if model is changed, this assumption may break
- **No batch support**: Single-image inference only
- **Class semantics undocumented**: The model output classes (indices 0-3) have no explicit labeling in code; only index 3 is interpreted as "good quality" in the test code

### Side Effect Warnings

- Model loading allocates GPU memory (if CUDA provider available) that persists for the object's lifetime
- No file system writes or global state mutations during inference

---

## 5. Vibe Coding Extension & Maintenance Guide

### Safe Modification Zones

| Zone | Safety | Notes |
|------|--------|-------|
| `__main__` block | Safe | Test/demo code only |
| Quality threshold logic | Safe | Changing `== 3` to another class index or adding confidence threshold |
| `detect` return format | Caution | Downstream consumers depend on the raw score array shape |
| Resize dimensions `(240, 320)` | Caution | Must match model's trained input resolution |

### Common Refactoring Pitfalls

- **Adding /255 normalization**: This model expects `[0, 255]` input; normalizing to `[0, 1]` will produce incorrect results
- **Changing resize dimensions**: Must match the model's expected spatial input exactly
- **Swapping width/height in resize**: `cv2.resize` takes `(width, height)`, so `(240, 320)` means width=240, height=320; reversing will produce wrong results

### Testing Recommendations

- **Unit test**: Provide a known good-quality image and verify argmax == 3; provide a known poor-quality image and verify argmax != 3
- **Shape test**: Verify model output shape matches expected number of classes
- **Edge case test**: Test with very small images, single-pixel images, and non-standard aspect ratios to ensure resize doesn't crash
