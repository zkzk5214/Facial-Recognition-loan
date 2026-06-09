import cv2
import time
import torch
import onnxruntime
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inferencer.general import non_max_suppression, scale_coords


class YoloDetection:
    def __init__(self, weights_add, device='cuda'):
        self.device = device
        self.weights_add = weights_add
        self.session = onnxruntime.InferenceSession(weights_add, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.imgsz = (640, 640)

    def letterbox(self, img, color=(114, 114, 114)):
        # Resize and pad image while meeting stride-multiple constraints
        shape = img.shape[:2]  # [height, width]

        # Scale ratio (new / old)
        r = min(self.imgsz[0] / shape[0], self.imgsz[1] / shape[1])

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = self.imgsz[1] - new_unpad[0], self.imgsz[0] - new_unpad[1]  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

        # add border CONSTANT with Gray color(default)
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

        img = img.transpose((2, 0, 1))[::-1]
        im = np.ascontiguousarray(img)
        return im

    def run_detect(self, input_image, conf_thres=0.3, iou_thres=0.45, max_det=1000, agnostic_nms=False):
        # Image Padding
        im = self.letterbox(input_image, self.imgsz)

        # Run inference
        im = torch.from_numpy(im).to(self.device).float()
        im /= 255
        im = im[None] if len(im.shape) == 3 else im  # [b, c, h, w]
        im = im.cpu().numpy()

        pred = self.session.run([self.session.get_outputs()[0].name], {self.session.get_inputs()[0].name: im})[0]
        # [b, 25500, 9]
        # 25500: 80*80*3+40*40*3+20*20*3+10*10*3, 9:obj4+conf1+cls4(face, eyes, mouth)
        pred = torch.tensor(pred) if isinstance(pred, np.ndarray) else pred

        pred = non_max_suppression(pred, conf_thres, iou_thres, max_det, agnostic_nms)

        # Process predictions
        results = torch.tensor([])
        for i, det in enumerate(pred):
            image_ = input_image.copy()
            if len(det):  # 8
                # Repadding
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], image_.shape).round()
                results = torch.cat((results, det), 0)  # [xyxy, conf, cls]

        return results.numpy()


if __name__ == '__main__':
    img = cv2.imread('./unit_test/test_img/1.jpg', cv2.IMREAD_COLOR)
    YD = YoloDetection(weights_add='./resources/arcface_weights_best_new.onnx')
    t = time.time()
    for _ in range(10):
        res = YD.run_detect(img)
    print(time.time() - t)
    print(res)
    