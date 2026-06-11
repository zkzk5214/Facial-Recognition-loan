import cv2
import torch
import torchvision
import onnxruntime
import numpy as np


def clip_coords(boxes, shape):
    """Clip xyxy bounding boxes to image boundaries (height, width)."""
    boxes[:, 0].clamp_(0, shape[1])
    boxes[:, 1].clamp_(0, shape[0])
    boxes[:, 2].clamp_(0, shape[1])
    boxes[:, 3].clamp_(0, shape[0])


def scale_coords(img1_shape, coords, img0_shape):
    """Rescale xyxy coords from letterboxed image (img1_shape) back to original image (img0_shape)."""
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
    pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    coords[:, [0, 2]] -= pad[0]  # remove x padding
    coords[:, [1, 3]] -= pad[1]  # remove y padding
    coords[:, :4] /= gain  # rescale to original size
    clip_coords(coords, img0_shape)
    return coords


def xywh2xyxy(x):
    """Convert boxes from [cx, cy, w, h] to [x1, y1, x2, y2] format."""
    y = x.clone()
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, max_det=300, agnostic=False):
    """
    Perform Non-Maximum Suppression on YOLO model output.

    Args:
        prediction: raw model output, shape (batch, num_anchors, 5+num_classes)
        conf_thres: objectness confidence threshold
        iou_thres: IoU threshold for NMS overlap removal
        max_det: maximum number of detections per image
        agnostic: if True, NMS is class-agnostic (no per-class offset)

    Returns:
        list of tensors (one per batch), each shape (n, 6) as [x1, y1, x2, y2, conf, cls]
    """
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    xc = prediction[..., 4] > conf_thres  # confidence candidates mask
    min_wh, max_wh = 2, 7680  # min/max box width and height (pixels)
    max_nms = 30000  # max boxes fed into torchvision.ops.nms

    output = [torch.zeros((0, 6), device=prediction.device) for _ in range(prediction.shape[0])]

    for xi, x in enumerate(prediction):  # iterate over batch
        # Zero out boxes with invalid width/height
        x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0
        x = x[xc[xi]]  # apply confidence filter

        if not x.shape[0]:
            continue

        # Compute class confidence = objectness * class_prob
        x[:, 5:] *= x[:, 4:5]
        box = xywh2xyxy(x[:, :4])

        # Build detections: [xyxy, best_conf, best_class_idx]
        conf, j = x[:, 5:].max(1, keepdim=True)
        x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        n = x.shape[0]
        if not n:
            continue
        elif n > max_nms:
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        # Batched NMS: offset boxes by class to prevent cross-class suppression
        c = x[:, 5:6] * (0 if agnostic else max_wh)
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)

        if i.shape[0] > max_det:
            i = i[:max_det]

        output[xi] = x[i]

    return output


class YoloDetection:
    def __init__(self, weights_add):
        self.session = onnxruntime.InferenceSession(
            weights_add,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.imgsz = (640, 640)

    def letterbox(self, img, color=(114, 114, 114)):
        """Resize and pad image while meeting stride-multiple constraints."""
        shape = img.shape[:2]

        r = min(self.imgsz[0] / shape[0], self.imgsz[1] / shape[1])

        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = self.imgsz[1] - new_unpad[0], self.imgsz[0] - new_unpad[1]

        dw /= 2
        dh /= 2

        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

        img = img.transpose((2, 0, 1))[::-1]
        return np.ascontiguousarray(img)

    def run_detect(self, input_image, conf_thres=0.3, iou_thres=0.45, max_det=1000, agnostic_nms=False):
        im = self.letterbox(input_image)
        im = im[None].astype(np.float32) / 255

        pred = self.session.run([self.output_name], {self.input_name: im})[0]
        pred = torch.tensor(pred) if isinstance(pred, np.ndarray) else pred
        pred = non_max_suppression(pred, conf_thres, iou_thres, max_det, agnostic_nms)

        dets = []
        for det in pred:
            if len(det):
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], input_image.shape).round()
                dets.append(det)

        return torch.cat(dets, 0).numpy() if dets else np.empty((0, 6))


if __name__ == '__main__':
    import time
    img = cv2.imread('./unit_test/test_img/1.jpg', cv2.IMREAD_COLOR)
    YD = YoloDetection(weights_add='./resources/arcface_weights_best_new.onnx')
    t = time.time()
    for _ in range(10):
        res = YD.run_detect(img)
    print(time.time() - t)
    print(res)
