import cv2
import onnxruntime
import numpy as np

model = onnxruntime.InferenceSession(
    './resources/maskModel.onnx',
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
_input_name = model.get_inputs()[0].name


def get_class(img):
    img_resized = cv2.resize(img, (224, 224))
    inputs = {_input_name: img_resized[None].astype(np.float32) / 255.0}
    res = model.run(None, inputs)[0][0]
    return np.argmax(res)


if __name__ == '__main__':
    img = cv2.imread('./unit_test/test_img/1.jpg', cv2.IMREAD_COLOR)
    print(get_class(img))
