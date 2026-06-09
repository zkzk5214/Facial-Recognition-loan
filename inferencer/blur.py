import cv2
import onnxruntime
import numpy as np
import torchvision.transforms as T
from PIL import Image


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0, keepdims=True)


class BlurDetection:
    def __init__(self, weights):
        self.model = onnxruntime.InferenceSession(weights, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    def featurize(self, images, transform):
        data = transform(images).unsqueeze(0).cpu().numpy()
        inputs = {self.model.get_inputs()[0].name: data}
        feature = self.model.run(None, inputs)[0][0]
        return softmax(feature)[0]

    def detect(self, img):
        transform = T.Compose([
            T.Resize((320, 240)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5]),
        ])
        img = Image.fromarray(img)
        features = self.featurize(img, transform)
        return features


if __name__ == '__main__':
    def detect_blur(img, thre=0.75, return_value=False):
        """Detect if the image is blurry."""

        features = bd.detect(img)

        if features <= thre:
            result = 0  # blur
        else:
            result = 1  # clear

        if return_value:
            return result, features
        else:
            return result
    
    bd = BlurDetection('./resources/blur_0727.onnx')

    img = cv2.imread('./resources/blur_image_1.jpg', cv2.IMREAD_COLOR)

    if img.shape[0] > 320:
        resize_high, resize_width = 320, int(img.shape[1] * 320 / img.shape[0])
        img = cv2.resize(img, (resize_width, resize_high))
    
    print(detect_blur(img, thre=0.75, return_value=True))
    