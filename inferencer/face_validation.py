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
    """
    results = []
    all_eyes = face_dict[1] + face_dict[2]
    for face in face_dict[0]:
        eye = check_in_face(all_eyes, face)
        eye = del_dup(eye)

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


if __name__ == '__main__':
    from collections import defaultdict

    face_Dict = defaultdict(list,
                            {0: [[148.0, 165.0, 363.0, 438.0]],
                             3: [[206.0, 344.0, 284.0, 386.0]],
                             2: [[176.0, 258.0, 228.0, 292.0]],
                             1: [[259.0, 257.0, 317.0, 290.0]]})
    for idx, is_complete in enumerate(complete_face(face_Dict)):
        print(is_complete)
        