import os
import sys
import cv2
import yaml
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inferencer.image_quality import ImgQuality
from inferencer.face_detector import Detector
from inferencer.face_pipeline import StatusCode

RESOURCE_PATH = './resources/'

with open('./config.yaml', 'r') as f:
    _config = yaml.safe_load(f)

_dark_bg_cfg = _config.get('dark_bg', {})
_model_threshold = float(_dark_bg_cfg.get('model_threshold', 0.95))
_dark_pixel_thresh = int(_dark_bg_cfg.get('dark_pixel_thresh', 50))
_dark_ratio_thresh = float(_dark_bg_cfg.get('dark_ratio_thresh', 0.75))
_bright_pixel_thresh = int(_dark_bg_cfg.get('bright_pixel_thresh', 150))
_bright_ratio_thresh = float(_dark_bg_cfg.get('bright_ratio_thresh', 0.1))
_bbox_expand_ratio = float(_dark_bg_cfg.get('bbox_expand_ratio', 0.0))
_bbox_expand_up_ratio = float(_dark_bg_cfg.get('bbox_expand_up_ratio', 0.1))

_quality_detector = None
_face_detector = None


def _init():
    global _quality_detector, _face_detector
    if _quality_detector is None:
        _quality_detector = ImgQuality(
            os.path.join(RESOURCE_PATH, 'weights-finetune-1-40--A.onnx'))
    if _face_detector is None:
        _face_detector = Detector(
            weights=os.path.join(RESOURCE_PATH, 'arcface_weights_best_new.onnx'),
            sp=os.path.join(RESOURCE_PATH, 'shape_predictor_68_face_landmarks.dat'))


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
        return False, -1.0, 0.0

    dark_ratio = (bg_pixels < _dark_pixel_thresh).mean()
    bright_ratio = (bg_pixels > _bright_pixel_thresh).mean()
    is_dark = bool(dark_ratio > _dark_ratio_thresh) and (bright_ratio < _bright_ratio_thresh)

    if is_dark:
        non_dark = bg_pixels[bg_pixels >= _dark_pixel_thresh]
        if len(non_dark) == 0:
            darkness_level = 1.0
        else:
            darkness_level = 1.0 - (float(non_dark.mean()) / 255.0)
        darkness_level *= 100.0
        darkness_level = round(darkness_level, 4)
    else:
        darkness_level = 0.0

    return bool(is_dark), float(dark_ratio), float(bright_ratio), darkness_level


def dark_bg_check(img_bgr):
    _init()

    scores = _quality_detector.detect(img_bgr)
    dark_score = float(scores[2])

    # Stage 1: model check
    stage1_pass = np.argmax(scores) == 2 and dark_score >= _model_threshold
    if not stage1_pass:
        return False, 100, dark_score, 0.0, 0.0, 0.0

    # Stage 2: CV background check
    face_bboxes = _face_detector.get_face_bboxes(img_bgr)
    if not face_bboxes:
        return False, StatusCode.NOFACE.value, dark_score, 0.0, 0.0, 0.0

    is_dark, dk_ratio, br_ratio, darkness_level = detect_bg_darkness(img_bgr, face_bboxes[0])

    if dk_ratio == -1.0:
        # background pixel count = 0, fall back to stage1 result
        return True, 100, dark_score, 0.0, 0.0, 0.0

    return is_dark, 100, dark_score, dk_ratio, br_ratio, darkness_level


if __name__ == '__main__':

    img_bgr = cv2.imread('./unit_test/test_img/black_bg_test.jpg')
    is_dark, resp_code, dark_score, dk_ratio, br_ratio, darkness_level = dark_bg_check(img_bgr)
    print(f'is_dark={is_dark}, resp_code={resp_code}, dark_score={dark_score:.4f}, dk_ratio={dk_ratio:.4f}, br_ratio={br_ratio:.4f}, darkness_level={darkness_level:.4f}')
