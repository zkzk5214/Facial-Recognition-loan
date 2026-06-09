import os
import sys
import dlib
import cv2
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inferencer.geo_check import complete_face
from inferencer.yolo_detection import YoloDetection


class Detector:
    def __init__(self, weights, sp):
        self.sp = dlib.shape_predictor(sp)  # face keypoint 68
        self.yolo_detect = YoloDetection(weights)

    def dlib_wrap(self, pic, x1, y1, x2, y2, size):
        img = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
        faces = dlib.full_object_detections()
        rec = dlib.rectangle(x1, y1, x2, y2)  # rectangle
        faces.append(self.sp(img, rec))
        # Face Align
        image = dlib.get_face_chip(img, (faces[0]), size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def get_face_capture(self, pic, size=112):  # 112 for TF-face
        # res = self.yolo_detect.run_detect(pic)  # [[x1,y1,x2,y2, conf, cls]...] cls:face,eyes,moth
        res = self.yolo_detect.run_detect(pic.copy())

        if len(res) <= 0:
            return [], []  # No BBX

        if len(res[res[:, 5] == 0]) == 0:
            return [], []  # No face(cls=0)

        face_conf = [int(i * 100) for i in res[res[:, 5] == 0][:, 4]]  # pick face conf
        face_chip_list = []

        face_dict = defaultdict(list)
        for i in res:
            face_dict[str(int(i[-1]))].append(list(i[:-2]))  # {cls: [x1,y1,x2,y2], [x1,y1,x2,y2]}

        for ii, boo_face in enumerate(complete_face(face_dict)):  # boo_face: True or False
            # bbx of face
            x1, y1 = int(face_dict['0'][ii][0]), int(face_dict['0'][ii][1])
            x2, y2 = int(face_dict['0'][ii][2]), int(face_dict['0'][ii][3])
            if not boo_face:
                continue  # skip incomplete face
            else:
                image = self.dlib_wrap(pic, x1, y1, x2, y2, size)
                face_chip_list.append(image)
        
        return face_chip_list, face_conf

if __name__ == '__main__':
    import time

    det = Detector(weights='./resources/arcface_weights_best_new.onnx', 
                   sp='./resources/shape_predictor_68_face_landmarks.dat')
    
    img = cv2.imread('./unit_test/test_img/register_test.jpg', cv2.IMREAD_COLOR)

    start = time.time()
    for _ in range(10):
        if img.shape[0] > 320:
            resize_high, resize_width = 320, int(img.shape[1] * 320 / img.shape[0])
            img = cv2.resize(img, (resize_width, resize_high))
        face_chip_list, face_conf = det.get_face_capture(img)
        print(len(face_chip_list), face_conf)
    time_cost = str(int((time.time() - start))/10)
    print(f'Average time cost: {time_cost} seconds')
