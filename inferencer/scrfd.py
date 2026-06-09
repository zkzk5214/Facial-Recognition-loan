import cv2
import datetime
import numpy as np
import onnxruntime


def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]

    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])

    return np.stack([x1, y1, x2, y2], axis=-1)


def preprocess(img, input_size):
    assert input_size is not None

    im_ratio = float(img.shape[0]) / img.shape[1]
    model_ratio = float(input_size[1]) / input_size[0]

    if im_ratio > model_ratio:
        new_height = input_size[1]
        new_width = int(new_height / im_ratio)
    else:
        new_width = input_size[0]
        new_height = int(new_width * im_ratio)

    det_scale = float(new_height) / img.shape[0]
    resized_img = cv2.resize(img, (new_width, new_height))
    det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
    det_img[:new_height, :new_width, :] = resized_img

    return det_img, det_scale


def postprocess(scores_list, bboxes_list, det_scale):
    # Sort Conf
    scores = np.vstack(scores_list)
    scores_ravel = scores.ravel()
    order = scores_ravel.argsort()[::-1]

    # Resize to ori scale
    bboxes = np.vstack(bboxes_list) / det_scale

    pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
    pre_det = pre_det[order, :]
    return pre_det


class SCRFD:
    def __init__(self, weights):
        self.session = onnxruntime.InferenceSession(
            weights,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

        self.batched = False
        self.center_cache = {}
        self._init_vars()

    def _init_vars(self):
        # 'input.1'
        self.input_name = self.session.get_inputs()[0].name
        # ['score_8', 'score_16', 'score_32', 'bbox_8', 'bbox_16', 'bbox_32']
        self.output_names = [o.name for o in self.session.get_outputs()]

        # self.input_size = None
        self.fmc = 3
        self._feat_stride_fpn = [8, 16, 32]
        self._num_anchors = 2
        self.nms_thresh = 0.6

    def forward(self, img, thresh):
        scores_list = []
        bboxes_list = []

        input_size = tuple(img.shape[0:2][::-1])  # input_size(320,320)
        blob = cv2.dnn.blobFromImage(img, 1.0 / 128, input_size, (127.5, 127.5, 127.5), swapRB=True)

        net_outs = self.session.run(self.output_names, {self.input_name: blob})

        input_height = blob.shape[2]  # 320
        input_width = blob.shape[3]   # 320
        fmc = self.fmc                # 3
        for idx, stride in enumerate(self._feat_stride_fpn):
            # If model support batch dim, take first output
            if self.batched:
                scores = net_outs[idx][0]
                bbox_preds = net_outs[idx + fmc][0]
                bbox_preds = bbox_preds * stride
                # If model doesn't support batching take output as is
            else:
                scores = net_outs[idx]  # (3200, 1)
                bbox_preds = net_outs[idx + fmc]
                bbox_preds = bbox_preds * stride  # (3200, 4)

            height = input_height // stride
            width = input_width // stride
            key = (height, width, stride)
            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)  # (40, 40, 2)
                anchor_centers = (anchor_centers * stride).reshape((-1, 2))  # (1600, 2)
                if self._num_anchors > 1:
                    anchor_centers = np.stack([anchor_centers] * self._num_anchors, axis=1).reshape(
                        (-1, 2))  # (3200, 2)
                if len(self.center_cache) < 100:
                    self.center_cache[key] = anchor_centers

            pos_inds = np.where(scores >= thresh)[0]
            bboxes = distance2bbox(anchor_centers, bbox_preds)
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)

        return scores_list, bboxes_list

    def nms(self, dets):
        x1, y1, x2, y2 = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h  
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= self.nms_thresh)[0]
            order = order[inds + 1]
        
        return keep
    
    def detect(self, img, thresh=0.5, input_size=(320, 320), max_num=0, metric='default'):
        
        # Image Resize and Padding to input_size
        det_img, det_scale = preprocess(img, input_size)

        # Detect
        scores_list, bboxes_list = self.forward(det_img, thresh)

        # Feature Resize and Concat
        pre_det = postprocess(scores_list, bboxes_list, det_scale)

        # NMS
        keep = self.nms(pre_det)
        det = pre_det[keep, :]

        # If candidate bbox is redundant
        if 0 < max_num < det.shape[0]:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            img_center = img.shape[0] // 2, img.shape[1] // 2
            offsets = np.vstack((
                (det[:, 0] + det[:, 2]) / 2 - img_center[1],
                (det[:, 1] + det[:, 3]) / 2 - img_center[0]
            ))
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            if metric == 'max':
                values = area
            else:
                values = area - offset_dist_squared * 2.0  # some extra weight on the centering
            bindex = np.argsort(
                values)[::-1]  # some extra weight on the centering
            bindex = bindex[0:max_num]
            det = det[bindex, :]

        return det


if __name__ == '__main__':
    
    detector = SCRFD(weights='./resources/scrfd.onnx')

    img = cv2.imread('./unit_test/test_img/1.jpg')
    if img.shape[0] > 320:
        resize_high, resize_width = 320, int(img.shape[1] / (img.shape[0] / 320))
        img = cv2.resize(img, [resize_high, resize_width])
        # img = cv2.resize(img, (240, 320)) # 320,240,3

    ta = datetime.datetime.now()
    for _ in range(100):
        bboxes = detector.detect(img, 0.05, input_size=(320, 320))
    tb = datetime.datetime.now()
    print('all cost:', (tb - ta).total_seconds() * 1000 / 100)
    print(bboxes)
