import os
import sys
import dlib
import cv2
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inferencer.face_validation import complete_face
from inferencer.yolo_detection import YoloDetection


class Detector:
    def __init__(self, weights, sp):
        self.sp = dlib.shape_predictor(sp)  # shape_predictor_68_face_landmarks.dat
        self.yolo_detect = YoloDetection(weights)

    def dlib_wrap(self, img_rgb, x1, y1, x2, y2, size):
        """Wrap dlib's face chip extraction."""
        rec = dlib.rectangle(x1, y1, x2, y2)
        shape = self.sp(img_rgb, rec)
        image = dlib.get_face_chip(img_rgb, shape, size)
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    def get_face_capture(self, pic, size=112):
        """Detect faces in the input image and return face chips and their confidence scores."""
        res = self.yolo_detect.run_detect(pic)

        if res.size == 0:
            return [], []
        # Filter out detected faces (class 0) and check if any faces are detected
        faces = res[res[:, 5] == 0]
        if len(faces) == 0:
            return [], []

        face_dict = defaultdict(list)
        for i in res:
            face_dict[int(i[-1])].append(list(i[:-2]))

        img_rgb = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
        face_chip_list = []
        face_conf = []

        for idx, is_complete in enumerate(complete_face(face_dict)):
            if not is_complete:
                continue
            x1, y1 = int(face_dict[0][idx][0]), int(face_dict[0][idx][1])
            x2, y2 = int(face_dict[0][idx][2]), int(face_dict[0][idx][3])
            face_chip_list.append(self.dlib_wrap(img_rgb, x1, y1, x2, y2, size))
            face_conf.append(int(faces[idx][4] * 100))

        return face_chip_list, face_conf

if __name__ == '__main__':
    import time

    det = Detector(weights='./resources/arcface_weights_best_new.onnx', 
                   sp='./resources/shape_predictor_68_face_landmarks.dat')
    
    img = cv2.imread('./unit_test/test_img/register_test.jpg', cv2.IMREAD_COLOR)

    if img.shape[0] > 320:
        resize_high, resize_width = 320, int(img.shape[1] * 320 / img.shape[0])
        img = cv2.resize(img, (resize_width, resize_high))

    start = time.time()
    for _ in range(10):
        face_chip_list, face_conf = det.get_face_capture(img)
        print(len(face_chip_list), face_conf)
    time_cost = str(int((time.time() - start)) / 10)
    print(f'Average time cost: {time_cost} seconds')
