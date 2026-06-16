import os
import cv2
import numpy as np
# import torchvision.transforms as T
# from enum import Enum
# from piq import total_variation
from sklearn.metrics.pairwise import cosine_similarity

from inferencer.face_detector import Detector
from inferencer.face_recognizer import Recognizer
# from inferencer.blur_detection import BlurDetection
# from inferencer.image_quality import ImgQuality


# class StatusCode(Enum):
#     SUCCESS = 100
#     BLURRED = 200
#     BACKLIGHT = 201
#     BLURRED_BACKLIGHT = 221
#     NOFACE = 300
#     MULTIFACE = 400
#     NO_FEATURE = 500
#     FEATURE_EXTRACT_ERROR = 600
#     ERROR = 999


# --- Image Preprocessing ---

def preprocess_image(img):
    """Resize large images preserving aspect ratio."""
    if img.shape[0] > 2400:
        resize_high, resize_width = 2400, int(img.shape[1] * 2400 / img.shape[0])
        return cv2.resize(img, (resize_width, resize_high))
    return img


# def calculate_image_brightness(img_bgr):
#     blue_avg, green_avg, red_avg, _ = cv2.mean(img_bgr)[:4]

#     # Rec. 709
#     weighted_terms = (
#         0.2126 * np.square(red_avg) +
#         0.7152 * np.square(green_avg) +
#         0.0722 * np.square(blue_avg)
#     )
#     brightness = np.sqrt(weighted_terms)

#     return max(min(brightness, 255), 0)


# def exposure_loss(img_bgr):
#     brightness = calculate_image_brightness(img_bgr)
#     mu = 127.5
#     scaling_denom = (mu ** 2) / 100
#     loss_val = ((brightness - mu) ** 2) / scaling_denom
#     return min(max(loss_val, 0), 100)


# --- Quality Detection Wrappers ---

# def detect_blur(img, threshold=0.75, return_value=False):
#     """Detect if the image is blurry."""
#     features = blur_detector.detect(img)
#     is_blurry = features > threshold
#     return (is_blurry, features) if return_value else is_blurry


# def detect_quality(img, return_value=False):
#     """Detect if the image has poor quality."""
#     features = quality_detector.detect(img)
#     result = (np.argmax(features, axis=-1) == 3)
#     return (result, features) if return_value else result


# def _determine_status(is_blurry, is_low_quality):
#     if is_blurry and is_low_quality:
#         return StatusCode.BLURRED_BACKLIGHT
#     if is_blurry:
#         return StatusCode.BLURRED
#     if is_low_quality:
#         return StatusCode.BACKLIGHT
#     return StatusCode.SUCCESS


# --- Feature Similarity ---

def get_topk(register_features, query_features, topk=3):
    """Get top K similar features and their indices."""
    similarity = (cosine_similarity(register_features, query_features) * 100).astype('int') - 3
    all_scores = np.max(similarity, axis=0)
    max_location = np.argsort(-all_scores).tolist()[:topk]
    max_score = [int(all_scores[i]) for i in max_location]
    return max_score


# --- Main Pipelines ---

def face_register(img, get_conf=True):
    processed_img = preprocess_image(img)
    face_chip_list, face_conf = face_detector.get_face_capture(processed_img)

    if get_conf:
        if len(face_chip_list) == 0:
            return [], []
        else:
            return [face_recognizer.recognize(chip) for chip in face_chip_list], face_conf
    else:
        if len(face_chip_list) == 0:
            return []
        else:
            return [face_recognizer.recognize(chip) for chip in face_chip_list]


# def head_detection_(img):
#     processed_img = preprocess_image(img)
#     face_chip_list, face_conf = face_detector.get_face_capture(processed_img)

#     face_count = len(face_chip_list)

#     if face_count == 0:
#         return 0, [], [], StatusCode.NOFACE.value

#     elif face_count > 1:
#         face_features = [face_recognizer.recognize(chip) for chip in face_chip_list]
#         return face_count, face_conf, face_features, StatusCode.MULTIFACE.value

#     is_blurry = detect_blur(img)
#     is_low_quality = detect_quality(img)

#     status_code = _determine_status(is_blurry, is_low_quality).value

#     face_features = [face_recognizer.recognize(chip) for chip in face_chip_list]

#     return face_count, face_conf, face_features, status_code


# --- Module Initialization ---

RESOURCE_PATH = './resources/'

# quality_detector = ImgQuality(os.path.join(RESOURCE_PATH, 'weights-finetune-l-40--A.onnx'))
# blur_detector = BlurDetection(weights=os.path.join(RESOURCE_PATH, 'blur_0727.onnx'))
face_detector = Detector(weights=os.path.join(RESOURCE_PATH, 'arcface_weights_best_new.onnx'),
                         sp=os.path.join(RESOURCE_PATH, 'shape_predictor_68_face_landmarks.dat'))
face_recognizer = Recognizer(weights=os.path.join(RESOURCE_PATH, 'TFace_pretrain.onnx'))

if __name__ == '__main__':

    imgDataRegisterCS = cv2.imread('./unit_test/test_img/2.jpg', cv2.IMREAD_COLOR)
    imgData = cv2.imread('./unit_test/test_img/3.jpg', cv2.IMREAD_COLOR)
    
    cs_feature = face_register(imgDataRegisterCS, get_conf=False)
    group_feature, detectionConf = face_register(imgData)
    print(get_topk(np.array(cs_feature), np.array(group_feature)))

