import cv2 as cv
import time
import numpy as np

rasp = False
if rasp:
    from tflite_runtime.interpreter import Interpreter
else:
    import tensorflow as tf


palette = {
    "black": (34, 40, 39),
    "pink": (114, 38, 249),
    "blue": (239, 217, 102),
    "green": (46, 226, 166),
    "orange": (31, 151, 253),
    "yellow": (102, 216, 255),
    "white": (193, 193, 193),
    "red": (0, 33, 255),
}


def draw_rect(image, box):
    h, w = image.shape[:2]
    y_min = int(max(1, (box[0] * h)))
    x_min = int(max(1, (box[1] * w)))
    y_max = int(min(h, (box[2] * h)))
    x_max = int(min(w, (box[3] * w)))

    # draw a rectangle on the image
    cv.rectangle(image, (x_min, y_min), (x_max, y_max), palette['blue'], 2)
    return image



PARTS = {
    0: 'NOSE',
    1: 'LEFT_EYE',
    2: 'RIGHT_EYE',
    3: 'LEFT_EAR',
    4: 'RIGHT_EAR',
    5: 'LEFT_SHOULDER',
    6: 'RIGHT_SHOULDER',
    7: 'LEFT_ELBOW',
    8: 'RIGHT_ELBOW',
    9: 'LEFT_WRIST',
    10: 'RIGHT_WRIST',
    11: 'LEFT_HIP',
    12: 'RIGHT_HIP',
    13: 'LEFT_KNEE',
    14: 'RIGHT_KNEE',
    15: 'LEFT_ANKLE',
    16: 'RIGHT_ANKLE'
}


class KeyPoint():
    def __init__(self, index, pos, v):
        x, y = pos
        self.x = x
        self.y = y
        self.index = index
        self.body_part = PARTS.get(index)
        self.confidence = v

    def point(self):
        return int(self.y), int(self.x)

    def to_string(self):
        return 'part: {} location: {} confidence: {}'.format(
            self.body_part, (self.x, self.y), self.confidence)


class Classifier:


    def __init__(self):

        model_name = 'posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite'

        if rasp:
            self.model = Interpreter(model_name)
        else:
            self.model = tf.lite.Interpreter(model_name)

        self.model.allocate_tensors()
        self.model_in = self.model.get_input_details()
        self.model_out = self.model.get_output_details()

        print(self.model_in)
        print(self.model_out)


    def get_keypoints(self, heatmaps, offsets, output_stride=32):
        scores = sigmoid(heatmaps)
        num_keypoints = scores.shape[2]
        heatmap_positions = []
        offset_vectors = []
        confidences = []
        for ki in range(0, num_keypoints):
            x, y = np.unravel_index(np.argmax(scores[:, :, ki]), scores[:, :, ki].shape)
            confidences.append(scores[x, y, ki])
            offset_vector = (offsets[y, x, ki], offsets[y, x, num_keypoints + ki])
            heatmap_positions.append((x, y))
            offset_vectors.append(offset_vector)
        image_positions = np.add(np.array(heatmap_positions) * output_stride, offset_vectors)
        keypoints = [KeyPoint(i, pos, confidences[i]) for i, pos in enumerate(image_positions)]
        return keypoints


    def predict(self, img_in):
        img_in = cv.resize(img_in, (257, 257)).astype('float32')
        img_out = img_in.copy()

        self.model.set_tensor(self.model_in[0]['index'], [img_in])
        self.model.invoke()

        # rects = self.model.get_tensor(
        #     self.model_out[0]['index'])
        # classes = self.model.get_tensor(
        #     self.model_out[1]['index'])
        # scores = self.model.get_tensor(
        #     self.model_out[2]['index'])
        #
        # for index, score in enumerate(scores[0]):
        #     if classes[0][index] == 0:
        #         if score > 0.5:
        #             img_out = draw_rect(img_in, rects[0][index])

        return img_out


clf = Classifier()
cap = cv.VideoCapture(0)
time.sleep(1.0)

while True:
    begin = time.time()
    ret, frame = cap.read()

    img_result = clf.predict(frame)
    # print(predicted)

    # show the frame
    cv.imshow("Frame", img_result)
    key = cv.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
    # print(f'took {np.round(time.time() - begin, 2)}seg')
cv.destroyAllWindows()
