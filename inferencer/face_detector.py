import dlib
import cv2
from collections import defaultdict
from inferencer.yolo_detection import YoloDetection


def check_in_face(bbox_list, face_bbox):
    """Check if the bounding boxes are within the face bounding box."""
    return [bbox for bbox in bbox_list
            if bbox[0] > face_bbox[0] and bbox[1] > face_bbox[1]
            and bbox[2] < face_bbox[2] and bbox[3] < face_bbox[3]]


def cal_iou(box1, box2):
    """Calculate the Intersection over Union (IoU) of two bounding boxes."""
    eps = 1e-7
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:4]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:4]
    inter = max(min(b1_x2, b2_x2) - max(b1_x1, b2_x1), 0) * max(min(b1_y2, b2_y2) - max(b1_y1, b2_y1), 0)
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union = w1 * h1 + w2 * h2 - inter + eps
    return inter / union


def del_dup(bbox_list, iou_thres=0.6):
    """Remove duplicate bounding boxes based on IoU threshold."""
    keep = set(range(len(bbox_list)))
    for i in range(len(bbox_list)):
        if i not in keep:
            continue
        for j in range(i + 1, len(bbox_list)):
            if j in keep and cal_iou(bbox_list[i], bbox_list[j]) > iou_thres:
                keep.discard(j)
    return [bbox_list[i] for i in sorted(keep)]


def check_intersection(box1, box2):
    """Check if two bounding boxes intersect."""
    return (min(box1[2], box2[2]) - max(box1[0], box2[0]) > 0 and
            min(box1[3], box2[3]) - max(box1[1], box2[1]) > 0)


def complete_face(face_dict):
    """
    Check if the detected components (eyes and mouth) are within the detected face bounding box
    and do not intersect with each other.
    person(0), left_eye(1), right_eye(2), mouth(3)
    """
    results = []
    # Combine all eyes from both left and right eye detections
    all_eyes = face_dict[1] + face_dict[2]
    for face in face_dict[0]:
        eye = check_in_face(all_eyes, face)
        eye = del_dup(eye)
        # Check if the mouth is within the face bounding box
        mouth = check_in_face(face_dict[3], face)
        if len(mouth) >= 2:
            mouth = del_dup(mouth)

        if len(eye) >= 2 and mouth:
            components = eye + mouth
            has_intersection = any(
                check_intersection(components[i], components[j])
                for i in range(len(components))
                for j in range(i + 1, len(components))
            )
            results.append(not has_intersection)
        else:
            results.append(False)

    return results


class Detector:
    def __init__(self, weights, sp):
        self.sp = dlib.shape_predictor(sp)
        self.yolo_detect = YoloDetection(weights)

    def dlib_wrap(self, img_rgb, x1, y1, x2, y2, size):
        rec = dlib.rectangle(x1, y1, x2, y2)
        shape = self.sp(img_rgb, rec)
        image = dlib.get_face_chip(img_rgb, shape, size)
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    def get_face_capture(self, pic, size=112):
        res = self.yolo_detect.run_detect(pic)

        if res.size == 0:
            return [], []
        
        # Filter out only the face detections (class 0)
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

    def get_face_bboxes(self, pic):
        res = self.yolo_detect.run_detect(pic)
        if res.size == 0:
            return []
        faces = res[res[:, 5] == 0]
        if len(faces) == 0:
            return []
        return faces[:, :4].astype(int).tolist()


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
