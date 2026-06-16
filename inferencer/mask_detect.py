import cv2
import onnxruntime
import numpy as np

def load_image(img):
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img.astype(np.float32), axis=0)
    img /= 255.0
    return img

def get_class(img):
    img = load_image(img)
    inputs = {model.get_inputs()[0].name: img}
    res = model.run(None, inputs)[0][0]
    return np.argmax(res, axis=1)

model = onnxruntime.InferenceSession(
    './resources/maskModel.onnx',
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

if __name__ == "__main__":
    img = cv2.imread('./unit_test/test_img/1.jpg', cv2.IMREAD_COLOR)
    print(get_class(img))
    