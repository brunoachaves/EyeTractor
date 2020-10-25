from imutils import face_utils
import numpy as np
import time
import dlib
import cv2 as cv

from azure.iot.device import IoTHubDeviceClient, Message
from datetime import datetime


pallete = {
    "red": (0, 33, 255),
    "blue": (255, 125, 0),
    "green": (61, 147, 77),
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "yellow": (0, 233, 255),
    "blue_bright": (189, 153, 0),
    "gray": (59, 48, 43),
}


# compute and return the euclidean distance between the two points
def euclidean_dist(pt_a, pt_b):
    return np.linalg.norm(pt_a - pt_b)


def distraction_ratio(list_eyes):
    if len(list_eyes) == 0:
        return None
    elif len(list_eyes) == 1:
        return 0
    else:
        return euclidean_dist(list_eyes[0], list_eyes[1])


def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    a = euclidean_dist(eye[1], eye[5])
    b = euclidean_dist(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    c = euclidean_dist(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (a + b) / (2.0 * c)

    return ear


# Metodo de monitoramento
# eventType pode ser 'normal' ou 'desatencao'
def send_message(eventType):
    CAMERAID = "camera-01"
    TODAY = datetime.now().isoformat()

    msg_txt_formatted = MSG_TXT.format(TODAY=TODAY, EVENT=eventType, CAMERAID=CAMERAID)
    message = Message(msg_txt_formatted)

    print("Enviando mensagem: {}".format(message))
    try:
        client.send_message(message)
        print("[Info] Mensagem enviada com sucesso")
    except:
        print("[Erro] Falha ao enviar a mensagem")


# Cria instancia do client do iot hub
CONNECTION_STRING = "HostName=eyetractor-hubiot.azure-devices.net;DeviceId=camera-01;SharedAccessKey=5pwN7TTPM7/V0bqXI+wEruGdCJi1o3h6QLEQRNwua+g="
client = IoTHubDeviceClient.create_from_connection_string(CONNECTION_STRING)
# Mensagem a ser enviada para o HubIot
MSG_TXT = '{{"CameraId": "{CAMERAID}", "EventType": "{EVENT}", "EventDate": "{TODAY}" }}'

EAR_THRESH = 0.3
EAR_CONSEC_FRAMES = 16
CONFIDENCE_FACE_DNN = 0.5  # minimum probability to filter weak detections
SUP_FACE = 0.7
DISTRACTION_THRESH = 1
DR_CONSEC_FRAMES = 16

MODEL_FILE = "models/res10_300x300_ssd_iter_140000.caffemodel"
PROTOTXT_FILE = "models/deploy.prototxt"

print("[INFO] loading models")
predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
face_cascade = cv.CascadeClassifier('models/haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('models/haarcascade_eye.xml')
net = cv.dnn.readNetFromCaffe(PROTOTXT_FILE, MODEL_FILE)

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


class VideoCamera(object):
    def __init__(self):
        print("[INFO] starting video stream thread...")
        self.cap = cv.VideoCapture(0)
        time.sleep(2.0)

        self.counter_ear = 0
        self.counter_dr = 0
        self.alarm_on = False
        self.ear = 0
        self.dr = 0
        self.clear_bar()
        self.status_color = ['red', 'green']

    def clear_bar(self):
        self.bar = np.ones((200, 800, 3)).astype('uint8')
        self.bar[..., 0] *= 98  # 59
        self.bar[..., 1] *= 50  # 48
        self.bar[..., 2] *= 35  # 43
        cv.putText(self.bar, "EYE TRACTOR", (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    '''
    def __del__(self):
        # releasing camera
        self.cap.release()
    '''

    def run(self, is_rasp=False):

        begin = time.time()
        ret, frame = self.cap.read()
        frame = cv.resize(frame, (400, 400))
        orig = frame.copy()
        self.clear_bar()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # detect faces in the grayscale frame
        rects = face_cascade.detectMultiScale(gray, scaleFactor=1.1,
                                              minNeighbors=5, minSize=(30, 30),
                                              flags=cv.CASCADE_SCALE_IMAGE)

        status = 0
        frontal_face_flag = False
        for (x, y, w, h) in rects:
            frontal_face_flag = True
            self.dr = 0
            rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
            cv.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 2)
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            # EAR average
            self.ear = (leftEAR + rightEAR) / 2.0

            leftEyeHull = cv.convexHull(leftEye)
            rightEyeHull = cv.convexHull(rightEye)
            cv.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            if self.ear < EAR_THRESH:
                self.counter_ear += 1
                if self.counter_ear >= EAR_CONSEC_FRAMES:
                    status = 1
                    if not self.alarm_on:
                        self.alarm_on = True
                        if is_rasp:
                            send_message("desatencao")
                            # Set buzzer and LEDs
                            GPIO.output(buzzer, GPIO.HIGH)
                            print("Beep")
                            None
            else:
                self.counter_ear = 0
                self.alarm_on = False
                if is_rasp:
                    GPIO.output(buzzer, GPIO.LOW)
                    print("no Beep")
                    # Reset buzzer and LEDs
                    None
            break

        if not frontal_face_flag:
            self.ear = 0
            (h, w) = frame.shape[:2]
            blob = cv.dnn.blobFromImage(cv.resize(frame, (300, 300)), 1.0,
                                        (300, 300), (104.0, 177.0, 123.0))
            net.setInput(blob)
            detections = net.forward()

            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence < CONFIDENCE_FACE_DNN:
                    continue

                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                cv.rectangle(frame, (startX, startY), (endX, endY),
                             (0, 0, 255), 2)

                face = frame[startY:startY + int((endY - startY) * SUP_FACE), startX:endX]
                face_gray = cv.cvtColor(face, cv.COLOR_BGR2GRAY)
                eyes = eye_cascade.detectMultiScale(face_gray, scaleFactor=1.1,
                                                    minNeighbors=5, minSize=(30, 30),
                                                    flags=cv.CASCADE_SCALE_IMAGE)

                eyes_pos_x = []
                for i, (ex, ey, ew, eh) in enumerate(eyes):
                    if i >= 2:
                        break
                    cv.rectangle(face, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                    eyes_pos_x.append(ex + (ew / 2))

                self.dr = distraction_ratio(eyes_pos_x)
                if self.dr is None or self.dr < DISTRACTION_THRESH:
                    self.counter_dr += 1
                    if self.counter_dr >= DR_CONSEC_FRAMES:
                        status = 2
                        if not self.alarm_on:
                            self.alarm_on = True
                            if is_rasp:
                                send_message("desatencao")
                                # Set buzzer and LEDs
                                GPIO.output(buzzer, GPIO.HIGH)
                                print("Beep")
                                None
                else:
                    self.counter_dr = 0
                    self.alarm_on = False
                    if is_rasp:
                        GPIO.output(buzzer, GPIO.LOW)
                        print("no Beep")
                        # Reset buzzer and LEDs
                        None

        if status == 0:
            cv.putText(self.bar, "MOTORISTA ATENTO", (10, 180),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, pallete['green'], 2)
        elif status == 1:
            cv.putText(self.bar, "ALERTA DE FADIGA!", (10, 180),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, pallete['red'], 2)
        elif status == 2:
            cv.putText(self.bar, "ALERTA DE DISTRACAO!", (10, 180),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, pallete['red'], 2)

        cv.putText(self.bar, f"EAR: {np.round(self.ear, 2)}", (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, pallete[self.status_color[status == 0]], 2)

        cv.putText(self.bar, f"DR: {self.dr}", (10, 130),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, pallete[self.status_color[status == 0]], 2)

        img_display = cv.hconcat((frame, orig))
        result_display = cv.vconcat((img_display, self.bar))
        print(f'loop time = {np.round(time.time() - begin, 2)}')
        return result_display


if __name__ == '__main__':
    vc = VideoCamera()
    while True:
        vc.get_frame()
