import math
from copy import deepcopy


def check_in_face(ele_list, face_xy):
    """check the whole bbx inside the face"""
    valid_ele = []

    for ele_xy in ele_list:
        if ele_xy[0] > face_xy[0] and ele_xy[1] > face_xy[1] and ele_xy[2] < face_xy[2] and ele_xy[3] < face_xy[3]:
            valid_ele.append(ele_xy)
        else:
            continue
    return valid_ele


def cal_iou(box1, box2, ciou=False, diou=False, giou=False):
    eps = 1e-7
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    inter = (max(min(b1_x2, b2_x2) - max(b1_x1, b2_x1), 0)) * (max(min(b1_y2, b2_y2) - max(b1_y1, b2_y1), 0))
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union
    if ciou or diou or giou:
        cw = max(b1_x2, b2_x2) - min(b1_x1, b2_x1)
        ch = max(b1_y2, b2_y2) - min(b1_y1, b2_y1)
        if ciou or diou:
            c2 = cw ** 2 + ch ** 2 + eps
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
            if ciou:
                v = (4 / math.pi ** 2) * pow(math.atan(w2 / h2) - math.atan(w1 / h1), 2)
                alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)
            return iou - rho2 / c2
        c_area = cw * ch + eps
        return iou - (c_area - union) / c_area
    return iou


def del_dup(ele_list, iou_thres=0.6):
    new_list = deepcopy(ele_list)
    for idx, bbx in enumerate(ele_list):
        for rem_bbx in ele_list[idx + 1:]:
            if (cal_iou(bbx, rem_bbx) > iou_thres) and (rem_bbx in new_list):
                new_list.remove(rem_bbx)
    return new_list


def check_intersection(box1, box2):
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    if min(b1_x2, b2_x2) - max(b1_x1, b2_x1) > 0 and min(b1_y2, b2_y2) - max(b1_y1, b2_y1) > 0:
        return True
    else:        
        return False


def complete_face(face_dict):
    boo_faces = []
    for face in face_dict[0]:
        eye = check_in_face(face_dict[1] + face_dict[2], face)
        eye = del_dup(eye)

        mouth = check_in_face(face_dict[3], face)
        if len(mouth) >= 2:
            mouth = del_dup(mouth)

        if len(eye) >= 2 and mouth:
            facial = eye + mouth
            for idx, bbx in enumerate(facial):
                for rem_bbx in facial[idx + 1:]:
                    if check_intersection(bbx, rem_bbx):
                        break
                else:  # no intersection
                    continue  # To the next idx,bbx
                break  # No more rem_bbx
            else:  # No break trigger in up loop
                boo_faces.append(True)
                continue  # To the next face
            boo_faces.append(False)

        else:  # Incomplete facial features
            boo_faces.append(False)

    return boo_faces


if __name__ == '__main__':
    from collections import defaultdict

    face_Dict = defaultdict(list,
                            {0: [[148.0, 165.0, 363.0, 438.0]],
                             3: [[206.0, 344.0, 284.0, 386.0]],
                             2: [[176.0, 258.0, 228.0, 292.0]],
                             1: [[259.0, 257.0, 317.0, 290.0]]})
    for ii, boo_face in enumerate(complete_face(face_Dict)):  # True or False
        print(boo_face)
        