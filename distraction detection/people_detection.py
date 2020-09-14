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


class Classifier:
    def __init__(self):
        model_name = 'coco_ssd_mobilenet_v1_1.0_quant_2018_06_29/detect.tflite'

        if rasp:
            self.model = Interpreter(model_name)
        else:
            self.model = tf.lite.Interpreter(model_name)

        self.model.allocate_tensors()
        self.model_in = self.model.get_input_details()
        self.model_out = self.model.get_output_details()

        print(self.model_in)
        print(self.model_out)

    def predict(self, img_in):
        img_in = cv.resize(img_in, (300, 300))
        img_out = img_in.copy()

        self.model.set_tensor(self.model_in[0]['index'], [img_in])
        self.model.invoke()

        rects = self.model.get_tensor(
            self.model_out[0]['index'])
        classes = self.model.get_tensor(
            self.model_out[1]['index'])
        scores = self.model.get_tensor(
            self.model_out[2]['index'])

        for index, score in enumerate(scores[0]):
            if classes[0][index] == 0:
                if score > 0.5:
                    img_out = draw_rect(img_in, rects[0][index])

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
    print(f'took {np.round(time.time() - begin, 2)}seg')
cv.destroyAllWindows()
