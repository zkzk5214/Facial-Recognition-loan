import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from inferencer.face_detector import Detector
from inferencer.face_recognizer import Recognizer


def preprocess_image(img):
    """Resize large images preserving aspect ratio."""
    if img.shape[0] > 2400:
        resize_high, resize_width = 2400, int(img.shape[1] * 2400 / img.shape[0])
        return cv2.resize(img, (resize_width, resize_high))
    return img


def get_topk(register_features, query_features, topk=3):
    """Get top K similarity scores between registered and query features."""
    similarity = (cosine_similarity(register_features, query_features) * 100).astype('int') - 3  # margin penalty
    all_scores = np.max(similarity, axis=0)
    max_location = np.argsort(-all_scores).tolist()[:topk]
    return [int(all_scores[i]) for i in max_location]


def face_register(img, get_conf=True):
    """Detect faces and extract feature embeddings."""
    processed_img = preprocess_image(img)
    face_chip_list, face_conf = face_detector.get_face_capture(processed_img)

    if len(face_chip_list) == 0:
        return ([], []) if get_conf else []

    features = [face_recognizer.recognize(chip) for chip in face_chip_list]
    return (features, face_conf) if get_conf else features


RESOURCE_PATH = './resources/'
face_detector = Detector(weights=f'{RESOURCE_PATH}arcface_weights_best_new.onnx',
                         sp=f'{RESOURCE_PATH}shape_predictor_68_face_landmarks.dat')
face_recognizer = Recognizer(weights=f'{RESOURCE_PATH}TFace_pretrain.onnx')

if __name__ == '__main__':

    imgDataRegisterCS = cv2.imread('./unit_test/test_img/2.jpg', cv2.IMREAD_COLOR)
    imgData = cv2.imread('./unit_test/test_img/3.jpg', cv2.IMREAD_COLOR)

    cs_feature = face_register(imgDataRegisterCS, get_conf=False)
    group_feature, detectionConf = face_register(imgData)
    print(get_topk(np.array(cs_feature), np.array(group_feature)))
