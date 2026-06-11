import onnxruntime
import numpy as np
import cv2


class ImgQuality:
    def __init__(self, weights):
        self.model = onnxruntime.InferenceSession(weights, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.input_name = self.model.get_inputs()[0].name

    def detect(self, img):
        img_resize = cv2.resize(img, (240, 320))
        inputs = {self.input_name: img_resize[None].astype(np.float32)}
        return self.model.run(None, inputs)[0][0]


if __name__ == '__main__':

    def detect_quality(img, return_value=False):
        res = iq.detect(img)
        result = (np.argmax(res, axis=-1) == 3)
        return (result, res) if return_value else result

    iq = ImgQuality('./resources/weights-finetune-1-40--A.onnx')

    img = cv2.imread('./unit_test/test_img/img_quality.jpg', cv2.IMREAD_COLOR)

    if img.shape[0] > 320:
        resize_high, resize_width = 320, int(img.shape[1] * 320 / img.shape[0])
        img = cv2.resize(img, (resize_width, resize_high))

    print(detect_quality(img, return_value=True))
    