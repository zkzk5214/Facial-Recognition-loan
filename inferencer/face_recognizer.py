import onnxruntime
import numpy as np
import torchvision.transforms as T


class Recognizer:
    def __init__(self, weights):
        self.model = onnxruntime.InferenceSession(weights, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.input_name = self.model.get_inputs()[0].name
        self.transform = T.Compose([
            T.ToPILImage(),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def recognize(self, img):
        data = self.transform(img)[None].cpu().numpy()
        feature = self.model.run(None, {self.input_name: data})[0][0]
        return feature / np.linalg.norm(feature)


if __name__ == '__main__':
    import cv2
    from sklearn.metrics.pairwise import cosine_similarity
    from inferencer.face_detector import Detector


    def get_topk(register, query, topk=3):
        similarity = (cosine_similarity(register, query) * 100).astype('int') - 3
        max_score = np.max(similarity, axis=0)
        max_location = np.argsort(-np.array(max_score)).tolist()[:topk]
        max_score = [int(max_score[i]) for i in max_location]
        return max_score, max_location, query[max_location].tolist()


    def face_register(img, get_sim=True):
        if img.shape[0] > 2400:
            # Geo-Scaling to [2400, X]
            resize_high, resize_width = 2400, int(img.shape[1] / (img.shape[0] / 2400))
            img = cv2.resize(img, [resize_width, resize_high])
        
        face_chip_list, face_similarities = det.get_face_capture(img)

        if get_sim:
            if len(face_chip_list) == 0:  # No face detected
                return [], []
            else:
                return [rec.recognize(face_chip) for face_chip in face_chip_list], face_similarities
        else:
            if len(face_chip_list) == 0:  # No face detected
                return []
            else:
                return [rec.recognize(face_chip) for face_chip in face_chip_list]
    

    det = Detector(weights='./resources/arcface_weights_best_new.onnx',
                   sp='./resources/shape_predictor_68_face_landmarks.dat')
    rec = Recognizer(weights='./resources/TFace_pretrain.onnx')

    imgDataRegisterCS = cv2.imread('./unit_test/test_img/2.jpg', cv2.IMREAD_COLOR)
    imgData = cv2.imread('./unit_test/test_img/3.jpg', cv2.IMREAD_COLOR)

    cs_feature = face_register(imgDataRegisterCS, get_sim=False)
    group_feature, detectionConf = face_register(imgData)

    print(get_topk(np.array(cs_feature), np.array(group_feature)))
    