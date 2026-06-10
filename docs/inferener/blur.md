# blur.py - Module Documentation

## 1. Module Overview

**Core Responsibility:** Provides image blur/sharpness detection capability using an ONNX-based classification model to determine whether an input image is blurry or clear.

**Key Functionalities:**
- Load and initialize an ONNX blur detection model with CUDA/CPU execution support
- Preprocess input images (resize, normalize) for model inference
- Run inference and return a sharpness confidence score via softmax classification

---

## 2. Architecture & Design Patterns

### Dependencies

| Library | Purpose |
|---------|---------|
| `cv2` (OpenCV) | Image I/O (used in `__main__` block) |
| `onnxruntime` | ONNX model inference engine |
| `numpy` | Numerical operations (softmax) |
| `torchvision.transforms` | Image preprocessing pipeline |
| `PIL.Image` | Image format conversion (ndarray -> PIL) |

### Design Patterns Applied

- **Encapsulation Pattern**: `BlurDetection` class encapsulates model loading, preprocessing, and inference into a single cohesive unit. The transform pipeline is initialized once and reused across calls.

### Data Flow

```
Input (np.ndarray BGR image)
    -> Convert to PIL Image
    -> Resize to (320, 240)
    -> ToTensor + Normalize
    -> ONNX model inference
    -> Softmax on raw logits
    -> Output: float (sharpness confidence score, 0~1)
```

---

## 3. Core Components Deep Dive

### `softmax(x)`

| Item | Detail |
|------|--------|
| **Purpose** | Convert raw model logits into probability distribution |
| **Parameters** | `x`: `np.ndarray` - raw output logits from the model |
| **Return** | `np.ndarray` - probability values summing to 1 |
| **Core Logic** | Applies the standard softmax formula `exp(x) / sum(exp(x))` along axis 0 |

> **Note:** Current implementation lacks numerical stability (no `x - max(x)` trick), which may cause overflow for large logit values.

---

### `BlurDetection.__init__(self, weights)`

| Item | Detail |
|------|--------|
| **Purpose** | Initialize the blur detection model and preprocessing pipeline |
| **Parameters** | `weights`: `str` - path to the `.onnx` model file |
| **Return** | None |
| **Core Logic** | Creates an ONNX inference session with CUDA preferred (CPU fallback); builds a fixed `torchvision.transforms` pipeline (Resize -> ToTensor -> Normalize) |

---

### `BlurDetection.detect(self, img)`

| Item | Detail |
|------|--------|
| **Purpose** | Run blur detection inference on a single image |
| **Parameters** | `img`: `np.ndarray` - input image in HWC format (typically BGR from OpenCV) |
| **Return** | `float` - sharpness confidence score; higher value = sharper image |
| **Core Logic** | Converts ndarray to PIL Image, applies transform pipeline, runs ONNX inference, and returns the softmax probability for the "sharp" class (index 0) |

---

## 4. Constraints, Edge Cases & Side Effects

### Preconditions

- The ONNX model file must exist at the specified `weights` path
- ONNX Runtime must be installed; for GPU acceleration, CUDA and cuDNN must be properly configured
- Input image must be a valid `np.ndarray` with 3 channels (HWC format)

### Known Limitations

- **Fixed input resolution**: All images are resized to `(320, 240)` regardless of aspect ratio, which may introduce distortion
- **Softmax numerical instability**: Large logit values may cause `np.exp` overflow
- **No batch inference**: Only single-image inference is supported; processing many images requires repeated calls
- **PIL conversion assumption**: `Image.fromarray(img)` assumes RGB or BGR uint8 input; non-standard dtype/channel images will fail silently or error

### Side Effect Warnings

- Model loading allocates GPU memory (if CUDA provider is available) which persists for the object's lifetime
- No file system writes or global state mutations

---

## 5. Vibe Coding Extension & Maintenance Guide

### Safe Modification Zones

| Zone | Safety | Notes |
|------|--------|-------|
| `__main__` block | Safe | Test/demo code only; no external consumers |
| `transform` pipeline in `__init__` | Moderate | Can adjust resize/normalize params, but must match model's expected input |
| `softmax` function | Safe | Can improve numerical stability without affecting API |
| `detect` method signature | Caution | External code depends on this interface |

### Common Refactoring Pitfalls

- **Changing normalize values**: The `mean=[0.5, 0.5, 0.5]` and `std=[0.5, 0.5, 0.5]` must match what the model was trained with; changing them will silently degrade accuracy
- **Altering resize dimensions**: `(320, 240)` is the model's expected input size; changing it will produce incorrect results
- **Removing `.cpu().numpy()`**: Even though inference is on CPU/GPU via ONNX Runtime, the tensor must be converted to numpy for the ONNX session input

### Testing Recommendations

- **Unit test**: Feed a known sharp image and verify score > 0.75; feed a known blurry image and verify score <= 0.75
- **Edge case test**: Test with very small images (e.g., 1x1), grayscale images, and non-uint8 arrays to verify graceful failure
- **Performance test**: Measure inference latency to ensure it meets real-time requirements if used in a pipeline
