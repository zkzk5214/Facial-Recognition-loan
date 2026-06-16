import dlib
import cv2
import numpy as np
from collections import defaultdict
from inferencer.yolo_detection import YoloDetection


def centre_bbx(bbox):
    """Return the center point (cx, cy) of a bounding box [x1, y1, x2, y2]."""
    return (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2


def check_in_face(bbox_list, face_bbox):
    """Filter bounding boxes whose center is inside the face bounding box."""
    result = []
    for bbox in bbox_list:
        cx, cy = centre_bbx(bbox)
        if face_bbox[0] < cx < face_bbox[2] and face_bbox[1] < cy < face_bbox[3]:
            result.append(bbox)
    return result


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


def vector_angle(v1, v2):
    """Calculate the angle (in degrees) between two 2D vectors."""
    # Calculate the angle using the arccosine of the dot product divided by the product of the magnitudes
    r = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    deg = np.degrees(r) # Convert radians to degrees
    # Determine the sign of the angle using the cross product
    if v1[0] * v2[1] - v1[1] * v2[0] > 0:
        deg = 360 - deg
    return deg


def get_rotation_matrix(p1, p2, x1, y1, x2, y2):
    """Compute rotation matrix to align eyes horizontally within the face bbox."""
    angle = vector_angle(p1 - p2, [1, 0])
    # Calculate the center of the face bounding box
    xc = (x1 + x2) // 2 
    yc = (y1 + y2) // 2
    # Get the rotation matrix for the specified angle and center
    M = cv2.getRotationMatrix2D((xc, yc), angle, 1)
    # Calculate the new width and height of the rotated image to ensure it fits without cropping
    cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
    nW = int((y2 * sin) + (x2 * cos)) # W = H*sin(θ) + W*cos(θ)
    nH = int((y2 * cos) + (x2 * sin)) # H = H*cos(θ) + W*sin(θ)
    # Shift the image to the center of the new dimensions
    M[0, 2] += (nW / 2) - xc 
    M[1, 2] += (nH / 2) - yc
    return M, angle


def calculate_eye_rotation(face, right_eyes, left_eyes):
    """Calculate rotation matrix and angle based on eye positions relative to face bbox."""
    fx, fy = face[0], face[1]
    # Calculate the relative positions of the eyes with respect to the face bounding box
    right_rel = [right_eyes[0]-fx, right_eyes[1]-fy, right_eyes[2]-fx, right_eyes[3]-fy]
    left_rel = [left_eyes[0]-fx, left_eyes[1]-fy, left_eyes[2]-fx, left_eyes[3]-fy]
    eye_right_center = np.array(centre_bbx(right_rel))
    eye_left_center = np.array(centre_bbx(left_rel))
    return get_rotation_matrix(eye_right_center, eye_left_center, 0, 0, face[2]-fx, face[3]-fy)


def cal_angle(point_a, point_b, point_c):
    """Calculate the signed angle at point_a between vectors AB and AC."""
    ab = np.array(point_b) - np.array(point_a)
    ac = np.array(point_c) - np.array(point_a)
    cos_phi = np.dot(ab, ac) / (np.linalg.norm(ab) * np.linalg.norm(ac))
    phi = np.arccos(cos_phi)
    if ab[0] * ac[1] - ac[0] * ab[1] < 0:
        return -phi
    return phi


def complete_face(face_dict):
    """
    Validate face completeness and compute rotation parameters.

    Returns:
        List of [is_complete: bool, rotation_params: list] for each face.
        rotation_params is [matrix, angle] if computable, else [].
    """
    results = []
    for face in face_dict['0']:
        eye = check_in_face(face_dict['1'] + face_dict['2'], face)
        eye = del_dup(eye)

        mouth = check_in_face(face_dict['3'], face)
        if len(mouth) >= 2:
            mouth = del_dup(mouth)

        if len(eye) >= 2 and mouth:
            components = eye + mouth
            has_intersection = any(
                check_intersection(components[i], components[j])
                for i in range(len(components))
                for j in range(i + 1, len(components))
            )
            if has_intersection:
                results.append([False, []])
            elif len(eye) == 2 and len(mouth) == 1:
                angle_ = cal_angle(centre_bbx(mouth[0]), centre_bbx(eye[0]), centre_bbx(eye[1]))
                right_eyes, left_eyes = (eye[1], eye[0]) if angle_ > 0 else (eye[0], eye[1])
                rotation_matrix, rotation_angle = calculate_eye_rotation(face, right_eyes, left_eyes)
                rotation_angle = 360 - rotation_angle if rotation_angle >= 180 else rotation_angle
                results.append([True, [rotation_matrix, rotation_angle]])
            else:
                results.append([True, []])
        else:
            results.append([False, []])

    return results


def _build_face_dict(res):
    """Group detection results by class label."""
    face_dict = defaultdict(list)
    for det in res:
        face_dict[str(int(det[-1]))].append(list(det[:-2]))
    return face_dict


def _get_bbox(face_dict, idx):
    """Extract integer bbox coordinates from face_dict."""
    b = face_dict['0'][idx]
    return int(b[0]), int(b[1]), int(b[2]), int(b[3])


class Detector:
    EXPAND_RATIO = 0.15

    def __init__(self, weights, sp):
        self.sp = dlib.shape_predictor(sp)
        self.yolo_detect = YoloDetection(weights)

    def _align_face(self, img_rgb, x1, y1, x2, y2, size):
        """Align a face using dlib's 68-landmark model. Expects RGB input."""
        rec = dlib.rectangle(x1, y1, x2, y2)
        shape = self.sp(img_rgb, rec)
        return dlib.get_face_chip(img_rgb, shape, size)

    def _rotate_and_align(self, img_rgb, rotate_para, x1, y1, x2, y2, size, angle_threshold):
        """Rotate face if needed, then align with dlib."""
        if rotate_para and rotate_para[1] > angle_threshold:
            rotation_matrix, rotation_angle = rotate_para
            sub_face = img_rgb[y1:y2, x1:x2, :]
            theta = rotation_angle * np.pi / 180
            new_w = int(abs(np.sin(theta) * sub_face.shape[0]) + abs(np.cos(theta) * sub_face.shape[1]))
            new_h = int(abs(np.sin(theta) * sub_face.shape[1]) + abs(np.cos(theta) * sub_face.shape[0]))
            sub_face = cv2.warpAffine(sub_face, rotation_matrix, (new_w, new_h), flags=cv2.INTER_CUBIC)
            return self._align_face(sub_face, 0, 0, new_w, new_h, size)

        return self._align_face(img_rgb, x1, y1, x2, y2, size)

    def _retry_incomplete_face(self, pic_rgb, x1, y1, x2, y2, size, angle_threshold):
        """Expand bbox and re-detect for incomplete faces. Returns aligned face or None."""
        h, w = pic_rgb.shape[:2]
        expand = self.EXPAND_RATIO
        new_x1 = max(0, int(x1 - expand * (x2 - x1)))
        new_y1 = max(0, int(y1 - expand * (y2 - y1)))
        new_x2 = min(w, int(x2 + expand * (x2 - x1)))
        new_y2 = min(h, int(y2 + expand * (y2 - y1)))

        new_pic = pic_rgb[new_y1:new_y2, new_x1:new_x2, :]
        new_pic_bgr = cv2.cvtColor(new_pic, cv2.COLOR_RGB2BGR)
        new_res = self.yolo_detect.run_detect(new_pic_bgr)

        if new_res.size == 0:
            return None
        if len(new_res[new_res[:, 5] == 0]) != 1:
            return None

        new_face_dict = _build_face_dict(new_res)
        new_face_status = complete_face(new_face_dict)[0]

        if not new_face_status[0]:
            return None

        nx1, ny1, nx2, ny2 = _get_bbox(new_face_dict, 0)
        return self._rotate_and_align(new_pic, new_face_status[1], nx1, ny1, nx2, ny2, size, angle_threshold)

    def get_face_capture(self, pic, angle_threshold=30, size=112):
        """Detect faces, validate completeness, rotate if needed, and return aligned face chips."""
        res = self.yolo_detect.run_detect(pic)

        if res.size == 0:
            return [], []

        faces = res[res[:, 5] == 0]
        if len(faces) == 0:
            return [], []

        face_dict = _build_face_dict(res)
        pic_rgb = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
        face_chip_list = []
        face_conf = []

        for idx, face_status in enumerate(complete_face(face_dict)):
            x1, y1, x2, y2 = _get_bbox(face_dict, idx)

            if not face_status[0]:
                image = self._retry_incomplete_face(pic_rgb, x1, y1, x2, y2, size, angle_threshold)
                if image is None:
                    continue
            else:
                image = self._rotate_and_align(pic_rgb, face_status[1], x1, y1, x2, y2, size, angle_threshold)

            face_chip_list.append(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            face_conf.append(int(faces[idx][4] * 100))

        return face_chip_list, face_conf


if __name__ == '__main__':

    det = Detector(weights='./resources/arcface_weights_best_new.onnx',
                   sp='./resources/shape_predictor_68_face_landmarks.dat')

    img = cv2.imread('./unit_test/test_img/3.jpg', cv2.IMREAD_COLOR)

    if img.shape[0] > 2400:
        resize_high, resize_width = 2400, int(img.shape[1] * 2400 / img.shape[0])
        img = cv2.resize(img, (resize_width, resize_high))

    face_chip_list, face_conf = det.get_face_capture(img)
    print(len(face_chip_list), face_conf)
