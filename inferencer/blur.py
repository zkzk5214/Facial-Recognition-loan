import onnxruntime
import numpy as np
import torchvision.transforms as T
from PIL import Image


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0, keepdims=True)


class BlurDetection:
    def __init__(self, weights):
        self.model = onnxruntime.InferenceSession(weights, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.transform = T.Compose([
            T.Resize((320, 240)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
     
    def detect(self, img):
        data = self.transform(Image.fromarray(img)).unsqueeze(0).cpu().numpy()
        inputs = {self.model.get_inputs()[0].name: data}
        feature = self.model.run(None, inputs)[0][0]
        return softmax(feature)[0]


if __name__ == '__main__':
    import cv2

    def detect_blur(img, thre=0.75, return_value=False):
        """Detect if the image is blurry."""

        features = bd.detect(img)
        result = int(features > thre)
        return (result, features) if return_value else result
    
    bd = BlurDetection('./resources/blur_0727.onnx')

    img = cv2.imread('./resources/blur_image_1.jpg', cv2.IMREAD_COLOR)

    if img.shape[0] > 320:
        resize_high, resize_width = 320, int(img.shape[1] * 320 / img.shape[0])
        img = cv2.resize(img, (resize_width, resize_high))
    
    print(detect_blur(img, thre=0.75, return_value=True))
    