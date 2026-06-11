# face_validation.py - Module Documentation

## 1. Module Overview

**Core Responsibility:** Provides geometric validation utilities to determine whether a detected face is "complete" — i.e., contains properly positioned, non-overlapping eyes and mouth within the face bounding box.

**Key Functionalities:**
- Filter bounding boxes that fall entirely within a parent bounding box
- Calculate IoU (Intersection over Union) between two bounding boxes
- Remove duplicate/overlapping bounding boxes via IoU-based suppression
- Check whether two bounding boxes intersect
- Validate face completeness by ensuring eyes and mouth are present and non-overlapping

---

## 2. Architecture & Design Patterns

### Dependencies

| Library | Purpose |
|---------|---------|
| *(None)* | Pure Python module with no external dependencies |

### Design Patterns Applied

- **Utility/Helper Pattern**: A collection of stateless pure functions with no side effects. Each function performs a single geometric operation.
- **Composition Pattern**: `complete_face` orchestrates the other utility functions into a validation pipeline.

### Data Flow

```
Input: face_dict {0: [face_bboxes], 1: [left_eye_bboxes], 2: [right_eye_bboxes], 3: [mouth_bboxes]}
    -> check_in_face: filter eyes/mouth that are inside the face bbox
    -> del_dup: remove overlapping duplicates (IoU > 0.6)
    -> Validate: len(eye) >= 2 AND mouth exists
    -> check_intersection: ensure no component overlaps another
    -> Output: List[bool] — per-face completeness flags
```

---

## 3. Core Components Deep Dive

### `check_in_face(bbox_list, face_bbox)`

| Item | Detail |
|------|--------|
| **Purpose** | Filter bounding boxes that are strictly contained within a parent face bbox |
| **Parameters** | `bbox_list`: `List[List[float]]` — candidate component bboxes `[x1,y1,x2,y2]`; `face_bbox`: `List[float]` — the enclosing face bbox |
| **Return** | `List[List[float]]` — subset of `bbox_list` fully inside `face_bbox` |
| **Core Logic** | Strict inequality check: all 4 edges of the child must be inside the parent (no touching) |

---

### `cal_iou(box1, box2)`

| Item | Detail |
|------|--------|
| **Purpose** | Compute standard IoU between two axis-aligned bounding boxes |
| **Parameters** | `box1`, `box2`: `List[float]` — bboxes in `[x1, y1, x2, y2]` format |
| **Return** | `float` — IoU value in `[0, 1]` |
| **Core Logic** | Computes intersection area via min/max clipping, computes union as `area1 + area2 - intersection + eps`, returns ratio. `eps=1e-7` on union prevents division by zero |

---

### `del_dup(bbox_list, iou_thres=0.6)`

| Item | Detail |
|------|--------|
| **Purpose** | Remove duplicate bounding boxes that overlap above an IoU threshold |
| **Parameters** | `bbox_list`: `List[List[float]]` — input bboxes; `iou_thres`: `float` — IoU threshold (default 0.6) |
| **Return** | `List[List[float]]` — deduplicated subset preserving original order |
| **Core Logic** | Greedy suppression: iterates pairs (i, j), discards j if IoU(i, j) > threshold. Uses a `set` for O(1) membership checks. Skips already-discarded indices |

---

### `check_intersection(box1, box2)`

| Item | Detail |
|------|--------|
| **Purpose** | Check if two bounding boxes have any overlapping area |
| **Parameters** | `box1`, `box2`: `List[float]` — bboxes in `[x1, y1, x2, y2]` format |
| **Return** | `bool` — `True` if boxes overlap |
| **Core Logic** | Tests if overlap width > 0 AND overlap height > 0 |

---

### `complete_face(face_dict)`

| Item | Detail |
|------|--------|
| **Purpose** | Validate whether each detected face has a complete set of non-overlapping facial components |
| **Parameters** | `face_dict`: `defaultdict(list)` with int keys — `{0: faces, 1: left_eyes, 2: right_eyes, 3: mouths}`, each value is a list of `[x1,y1,x2,y2]` |
| **Return** | `List[bool]` — one boolean per face; `True` = complete, `False` = incomplete or has overlapping components |
| **Core Logic** | For each face: filters eyes/mouth inside face bbox → deduplicates → requires >= 2 eyes and >= 1 mouth → checks all component pairs for intersection → appends `True` only if all conditions pass |

---

## 4. Constraints, Edge Cases & Side Effects

### Preconditions

- `face_dict` must be a `defaultdict(list)` (or equivalent) so that accessing missing keys returns `[]` instead of raising `KeyError`
- Bounding boxes must be in `[x1, y1, x2, y2]` format where `x2 > x1` and `y2 > y1`
- Keys must be integers: `0` (face), `1` (left eye), `2` (right eye), `3` (mouth)

### Known Limitations

- **Strict containment in `check_in_face`**: Uses `>` / `<` (not `>=` / `<=`), so components touching the face edge exactly are excluded
- **No confidence-based ordering in `del_dup`**: Greedy suppression keeps the first bbox by index, not necessarily the highest-confidence one
- **O(n^2) complexity**: Both `del_dup` and `check_intersection` in `complete_face` use pairwise comparisons; acceptable for small N (typically < 10 components per face)
- **Assumes well-formed boxes**: No validation that `x2 > x1` or `y2 > y1`; negative-area boxes produce undefined IoU results

### Side Effect Warnings

- Pure functions with no side effects
- No I/O, no global state mutation, no memory allocation beyond return values

---

## 5. Vibe Coding Extension & Maintenance Guide

### Safe Modification Zones

| Zone | Safety | Notes |
|------|--------|-------|
| `iou_thres` default (0.6) | Safe | Tunable; higher = less aggressive dedup |
| `check_in_face` inequality operators | Moderate | Changing `>` to `>=` includes edge-touching components; may affect downstream logic |
| `complete_face` validation rules | Moderate | e.g., requiring >= 1 eye instead of >= 2 loosens the filter |
| `__main__` block | Safe | Test code only |

### Common Refactoring Pitfalls

- **Changing `face_dict` key scheme**: Must stay consistent with `detector.py` which builds the dict with int keys `0, 1, 2, 3`
- **Reordering `del_dup` logic**: The greedy left-to-right suppression means insertion order matters; sorting by confidence first would change which boxes survive
- **Adding `<=` to `check_in_face`**: May cause components on the face boundary to pass validation, potentially including false positives

### Testing Recommendations

- **Unit test `cal_iou`**: Test identical boxes (IoU=1.0), non-overlapping boxes (IoU=0.0), partial overlap, and zero-area boxes
- **Unit test `complete_face`**: Provide faces with known complete/incomplete component layouts; verify boolean outputs
- **Edge case test**: Empty `face_dict` (all keys map to `[]`), single face with no eyes/mouth, overlapping duplicate components
