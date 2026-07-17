import cv2
import yaml
import numpy as np

from inferencer.face_pipeline import quality_detector, face_detector, StatusCode

with open('./config.yaml', 'r') as f:
    _config = yaml.safe_load(f)

_dark_bg_cfg = _config.get('dark_bg', {})
_model_threshold = float(_dark_bg_cfg.get('model_threshold', 0.95))
_dark_pixel_thresh = int(_dark_bg_cfg.get('dark_pixel_thresh', 50))
_dark_ratio_thresh = float(_dark_bg_cfg.get('dark_ratio_thresh', 0.75))
_bright_thresh = int(_dark_bg_cfg.get('bright_thresh', 80))
_min_patch_area = int(_dark_bg_cfg.get('min_patch_area', 200))
_patch_ratio_thresh = float(_dark_bg_cfg.get('patch_ratio_thresh', 0.03))
_spot_ratio_thresh = float(_dark_bg_cfg.get('spot_ratio_thresh', 0.10))
_bbox_expand_ratio = float(_dark_bg_cfg.get('bbox_expand_ratio', 0.0))
_bbox_expand_up_ratio = float(_dark_bg_cfg.get('bbox_expand_up_ratio', 0.1))

def preprocess_image(img):
    if img.shape[0] > 2400:
        resize_high, resize_width = 2400, int(img.shape[1] * 2400 / img.shape[0])
        return cv2.resize(img, (resize_width, resize_high))
    return img


def classify_bright_regions(gray, mask, bright_thresh, min_patch_area):
    bg_only = gray.copy()
    bg_only[mask != 255] = 0
    _, binary = cv2.threshold(bg_only, bright_thresh, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    spot_area = 0
    patch_area = 0

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_patch_area:
            spot_area += area
        else:
            patch_area += area

    total = max(len(gray[mask == 255]), 1)
    return spot_area / total, patch_area / total


def detect_bg_darkness(img_bgr, face_bbox):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    x1, y1, x2, y2 = face_bbox

    bbox_h = y2 - y1
    bbox_w = x2 - x1

    expand_up = int(bbox_h * _bbox_expand_up_ratio)
    expand_other = int(min(bbox_h, bbox_w) * _bbox_expand_ratio)

    x1 = max(0, x1 - expand_other)
    y1 = max(0, y1 - expand_up)
    x2 = min(w, x2 + expand_other)
    y2 = min(h, y2 + expand_other)

    mask = np.ones((h, w), dtype=np.uint8) * 255
    mask[y1:y2, x1:x2] = 0
    mask[y2:, :] = 0

    bg_pixels = gray[mask == 255]
    if len(bg_pixels) == 0:
        return False, -1.0, 0.0, 0.0, 0.0

    dark_ratio = (bg_pixels < _dark_pixel_thresh).mean()
    spot_ratio, patch_ratio = classify_bright_regions(
        gray, mask, _bright_thresh, _min_patch_area)
    is_dark = bool(dark_ratio > _dark_ratio_thresh) \
        and patch_ratio < _patch_ratio_thresh \
        and spot_ratio < _spot_ratio_thresh

    if is_dark:
        non_dark = bg_pixels[bg_pixels >= _dark_pixel_thresh]
        if len(non_dark) == 0:
            darkness_level = 1.0
        else:
            darkness_level = 1.0 - (float(non_dark.mean()) / 255.0)
        darkness_level = round(darkness_level, 4)
    else:
        darkness_level = 0.0

    return bool(is_dark), float(dark_ratio), float(spot_ratio), float(patch_ratio), darkness_level


def dark_bg_check(img_bgr):
    img_bgr = preprocess_image(img_bgr)
    scores = quality_detector.detect(img_bgr)
    dark_score = float(scores[2])

    # Stage 1: model check
    stage1_pass = np.argmax(scores) == 2 and dark_score >= _model_threshold
    if not stage1_pass:
        return False, 100, dark_score, 0.0, 0.0, 0.0, 0.0

    # Stage 2: CV background check
    face_bboxes = face_detector.get_face_bboxes(img_bgr)
    if not face_bboxes:
        return False, StatusCode.NOFACE.value, dark_score, 0.0, 0.0, 0.0, 0.0

    is_dark, dk_ratio, spot_ratio, patch_ratio, darkness_level = detect_bg_darkness(img_bgr, face_bboxes[0])

    if dk_ratio == -1.0:
        return True, 100, dark_score, 0.0, 0.0, 0.0, 0.0

    return is_dark, 100, dark_score, dk_ratio, spot_ratio, patch_ratio, darkness_level


if __name__ == '__main__':

    img_bgr = cv2.imread('./unit_test/test_img/black_bg_test.jpg')
    is_dark, resp_code, dark_score, dk_ratio, spot_ratio, patch_ratio, darkness_level = dark_bg_check(img_bgr)
    print(f'is_dark={is_dark}, resp_code={resp_code}, dark_score={dark_score:.4f}, dk_ratio={dk_ratio:.4f}, spot_ratio={spot_ratio:.4f}, patch_ratio={patch_ratio:.4f}, darkness_level={darkness_level:.4f}')
