# face_recognizer.py - Module Documentation

## 1. Module Overview

**Core Responsibility:** Extracts L2-normalized face embedding vectors from aligned face images using an ONNX recognition model (TFace), enabling downstream face similarity comparison.

**Key Functionalities:**
- Load and initialize an ONNX face recognition model with CUDA/CPU support
- Preprocess aligned face chips (PIL conversion, normalization) for model input
- Run inference and return L2-normalized 512-dimensional feature vectors

---

## 2. Architecture & Design Patterns

### Dependencies

| Library | Purpose |
|---------|---------|
| `onnxruntime` | ONNX model inference engine |
| `numpy` | L2 normalization and array operations |
| `torchvision.transforms` | Image preprocessing pipeline (ToPILImage, ToTensor, Normalize) |

### Design Patterns Applied

- **Encapsulation Pattern**: `Recognizer` class wraps model loading, transform initialization, and inference into a single `recognize` method.
- **Eager Initialization**: Transform pipeline and ONNX input name are created once in `__init__` and reused across all calls.

### Data Flow

```
Input (np.ndarray, aligned face chip, HWC, BGR, uint8, typically 112x112)
    -> ToPILImage: ndarray -> PIL Image
    -> ToTensor: PIL -> float32 tensor [0, 1]
    -> Normalize: (x - 0.5) / 0.5 -> [-1, 1]
    -> Add batch dim + convert to numpy
    -> ONNX inference: extract raw embedding
    -> L2 normalize
    -> Output: np.ndarray (512-d unit vector)
```

---

## 3. Core Components Deep Dive

### `Recognizer.__init__(self, weights)`

| Item | Detail |
|------|--------|
| **Purpose** | Initialize ONNX model, cache input name, and build transform pipeline |
| **Parameters** | `weights`: `str` - path to ONNX recognition model (e.g., TFace_pretrain.onnx) |
| **Return** | None |
| **Core Logic** | Creates ONNX InferenceSession with CUDA/CPU providers; caches `self.input_name`; builds a fixed transform: ToPILImage → ToTensor → Normalize(0.5, 0.5) |

---

### `Recognizer.recognize(self, img)`

| Item | Detail |
|------|--------|
| **Purpose** | Extract L2-normalized face embedding from an aligned face image |
| **Parameters** | `img`: `np.ndarray` - aligned face chip (HWC, BGR/RGB, uint8, typically 112x112) |
| **Return** | `np.ndarray` - 1D float32 array of shape `(512,)`, L2-normalized (unit vector) |
| **Core Logic** | Applies transform pipeline to convert image to normalized tensor, adds batch dimension, runs ONNX inference to get raw embedding, divides by L2 norm to produce unit vector suitable for cosine similarity |

---

## 4. Constraints, Edge Cases & Side Effects

### Preconditions

- ONNX model file must exist at the specified path and output a 512-d embedding vector
- Input image should be an aligned face chip (output of `face_detector.Detector.dlib_wrap`), typically 112x112 pixels
- Input must be a valid `np.ndarray` with 3 channels (HWC, uint8)

### Known Limitations

- **No batch inference**: Processes one face chip at a time; multiple faces require repeated calls
- **Fixed normalization assumption**: `mean=0.5, std=0.5` maps [0,1] to [-1,1]; must match model's training preprocessing
- **No input size validation**: Does not verify the image is 112x112; wrong sizes may produce degraded embeddings silently
- **Zero-norm edge case**: If model outputs an all-zero vector, `np.linalg.norm` returns 0, causing division by zero (NaN output)

### Side Effect Warnings

- Model loading allocates GPU memory (if CUDA provider available) that persists for the object's lifetime
- No file system writes or global state mutations
- `ToPILImage()` assumes the input is BGR uint8; incorrect input types may produce silent color channel errors

---

## 5. Vibe Coding Extension & Maintenance Guide

### Safe Modification Zones

| Zone | Safety | Notes |
|------|--------|-------|
| `__main__` block | Safe | Test/demo code only |
| L2 normalization logic | Safe | Can add epsilon for zero-norm protection |
| Transform pipeline order | Caution | Must match model's trained preprocessing exactly |
| Output dimensionality | Caution | Downstream `cosine_similarity` assumes consistent vector size |

### Common Refactoring Pitfalls

- **Changing normalize mean/std**: Must match the model's training configuration exactly; mismatches produce poor embeddings
- **Removing `ToPILImage`**: `ToTensor` requires PIL input to correctly scale uint8 [0,255] → float32 [0,1]; removing it breaks the pipeline
- **Adding resize to transform**: The model expects a specific input size (112x112); resizing here would conflict with the upstream `dlib_wrap` which already handles alignment and sizing
- **Switching to batch inference**: Requires changing the transform to handle a list/batch of images and modifying the ONNX session call

### Testing Recommendations

- **Identity test**: Feed the same face image twice; verify cosine similarity ≈ 1.0
- **Different faces test**: Feed two different people's faces; verify cosine similarity < 0.5
- **Unit vector test**: Verify `np.linalg.norm(output)` ≈ 1.0 for any input
- **Determinism test**: Feed the same image multiple times; verify outputs are identical
