import os
import sys
import cv2
import numpy as np
import torchvision.transforms as T
from enum import Enum
from piq import total_variation  #### New pip
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inferencer.detector import Detector
from inferencer.recognizer import Recognizer
from inferencer.blur import BlurDetection
from inferencer.quality import ImgQuality


class StatusCode(Enum):
    SUCCESS = 100
    BLURRED = 200
    BACKLIGHT = 201
    BLURRED_BACKLIGHT = 221
    NOFACE = 300
    MULTIFACE = 400
    NO_FEATURE = 500
    FEATURE_EXTRACT_ERROR = 600
    ERROR = 999


def preprocess_image(img):
    """Resize large images preserving aspect ratio."""
    if img.shape[0] > 2400:
        resize_high, resize_width = 2400, int(img.shape[1] / (img.shape[0] / 2400))
        return cv2.resize(img, (resize_width, resize_high))
    return img


def detect_blur(img, thre=0.75, return_value=False):
    """Detect if the image is blurry."""
    features = bd.detect(img)
    result = 0 if features <= thre else 1
    return (result, features) if return_value else result


def detect_quality(img, return_value=False):
    """Detect if the image has poor quality."""
    features = iq.detect(img)
    result = (np.argmax(features, axis=-1) == 3)
    return (result, features) if return_value else result


def calculate_image_brightness(img_bgr):
    blue_avg, green_avg, red_avg, _ = cv2.mean(img_bgr)[:4]

    # Rec. 709
    weighted_terms = (
        0.2126 * np.square(red_avg) +
        0.7152 * np.square(green_avg) +
        0.0722 * np.square(blue_avg)
    )
    brightness = np.sqrt(weighted_terms)

    return max(min(brightness, 255), 0)


def exposure_loss(img_bgr):
    brightness = calculate_image_brightness(img_bgr)
    mu = 127.5
    scaling_denom = (mu ** 2) / 100
    loss_val = ((brightness - mu) ** 2) / scaling_denom
    return min(max(loss_val, 0), 100)


def get_topk(query_1, query_2, topk=3):
    """Get top K similar features and their indices."""
    # Cal sim b2 feature and group features
    similarity = (cosine_similarity(query_1, query_2) * 100).astype('int')
    max_score = np.max(similarity, axis=0)
    max_location = np.argsort(-np.array(max_score)).tolist()[:topk]
    max_score = [int(max_score[i]) for i in max_location]
    return max_score, max_location, query_2[max_location].tolist()


def face_register(img):
    processed_img = preprocess_image(img)
    face_chip_list, _ = det.get_face_capture(processed_img)

    face_count = len(face_chip_list)

    if face_count == 0:  # No-face
        return [], StatusCode.NOFACE.value, 0.0

    elif face_count > 1:  # Multi-face
        face_features = [rec.recognize(chip) for chip in face_chip_list]
        return face_features, StatusCode.MULTIFACE.value, 0.0

    is_blurry_full, blur_score = detect_blur(img, return_value=True)  # Blur Detect
    total_variation_score = 100 - total_variation(T.ToTensor()(img).unsqueeze(0)).tolist()
    rms_score = exposure_loss(img) * 0.1

    single_face = face_chip_list[0]
    is_low_quality, light_score = detect_quality(single_face, return_value=True)  # Quality Detect
    light_de_dark = light_score[2] * 100
    light_de_bright = light_score[3] * 100

    image_quality_score = 100 - np.max(
        [blur_score * 100, total_variation_score, light_de_dark, light_de_bright]) - rms_score
    image_quality_score = round(min(max(image_quality_score, 1), 100), 4)

    status_conditions = [
        (is_blurry_full and is_low_quality, StatusCode.BLURRED_BACKLIGHT),
        (is_blurry_full, StatusCode.BLURRED),
        (is_low_quality, StatusCode.BACKLIGHT),
        (True, StatusCode.SUCCESS)    # Default
    ]
    status_code = next(code for cond, code in status_conditions if cond).value

    face_features = [rec.recognize(chip) for chip in face_chip_list]

    return face_features, status_code, image_quality_score

def head_detection_(img):
    processed_img = preprocess_image(img)
    face_chip_list, face_conf = det.get_face_capture(processed_img)

    face_count = len(face_chip_list)

    if face_count == 0:  # No-face
        return 0, [], [], StatusCode.NOFACE.value

    elif face_count > 1:  # Multi-face
        face_features = [rec.recognize(chip) for chip in face_chip_list]
        return face_count, face_conf, face_features, StatusCode.MULTIFACE.value

    is_blurry = detect_blur(img)    # Blur Detect
    is_low_quality = detect_quality(img)    # Quality Detect

    status_conditions = [
        (is_blurry and is_low_quality, StatusCode.BLURRED_BACKLIGHT),
        (is_blurry, StatusCode.BLURRED),
        (is_low_quality, StatusCode.BACKLIGHT),
        (True, StatusCode.SUCCESS)    # Default
    ]

    status_code = next(code for cond, code in status_conditions if cond).value

    face_features = [rec.recognize(chip) for chip in face_chip_list]

    return face_count, face_conf, face_features, status_code


# Initialize detectors and recognizers with resource paths
RESOURCE_PATH = './resources/'

iq = ImgQuality(os.path.join(RESOURCE_PATH, 'weights-finetune-l-40--A.onnx'))
bd = BlurDetection(weights=os.path.join(RESOURCE_PATH, 'blur_0727.onnx'))
det = Detector(weights=os.path.join(RESOURCE_PATH, 'arcface_weights_best_new.onnx'),
               sp=os.path.join(RESOURCE_PATH, 'shape_predictor_68_face_landmarks.dat'))
rec = Recognizer(weights=os.path.join(RESOURCE_PATH, 'TFace_pretrain.onnx'))

if __name__ == '__main__':

    imgDataRegister = cv2.imread('./unit_test/test_img/register_test.jpg', cv2.IMREAD_COLOR)
    for _ in range(10):
        face_features, status_code, quality_score = face_register(imgDataRegister)
        if status_code == StatusCode.SUCCESS.value:
            max_score, _ , vector = get_topk(np.array(face_features), np.array(face_features))
            print(max_score, len(vector[0]))
        else:
            print(f"Status Code: {status_code}, Quality Score: {quality_score}")