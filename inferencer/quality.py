import onnxruntime
import numpy as np
import cv2


class ImgQuality:
    def __init__(self, weights):
        self.model = onnxruntime.InferenceSession(weights, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    def detect(self, img):
        img_resize = cv2.resize(img, [240, 320])
        inputs = {self.model.get_inputs()[0].name: np.array([img_resize.astype('float32')])}
        res = self.model.run(None, inputs)[0][0]
        return res


if __name__ == '__main__':

    def detect_quality(img, return_value=False):
        """Detect if the image has poor quality."""

        res = iq.detect(img)
        result = (np.argmax(res, axis=-1) == 3)

        if return_value:
            return result, res
        else:
            return result


    iq = ImgQuality('./resources/weights-finetune-1-40--A.onnx')

    img = cv2.imread('./unit_test/test_img/img_quality.jpg', cv2.IMREAD_COLOR)

    if img.shape[0] > 320:
        # Geo-Scaling to [320, X]
        resize_high, resize_width = 320, int(img.shape[1] / (img.shape[0] / 320))
        img = cv2.resize(img, [resize_width, resize_high])

    print(detect_quality(img, return_value=True))  # 1, 2, 3
    