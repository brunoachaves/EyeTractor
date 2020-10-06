from imutils import face_utils
import numpy as np
import argparse
import time
import dlib
import cv2 as cv
from azure.iot.device import IoTHubDeviceClient, Message
from datetime import datetime


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
    client.send_message(message)
    print("Mensagem enviada com sucesso")


# Cria instancia do client do iot hub
CONNECTION_STRING = "HostName=eyetractor-hubiot.azure-devices.net;DeviceId=camera-01;SharedAccessKey=5pwN7TTPM7/V0bqXI+wEruGdCJi1o3h6QLEQRNwua+g="
client = IoTHubDeviceClient.create_from_connection_string(CONNECTION_STRING)
# Mensagem a ser enviada para o HubIot
MSG_TXT = '{{"CameraId": "{CAMERAID}", "EventType": "{EVENT}", "EventDate": "{TODAY}" }}'

ap = argparse.ArgumentParser()
ap.add_argument("-a", "--alarm", type=int, default=0,
                help="boolean used to indicate if Raspberry Pi IO should be used")
args = vars(ap.parse_args())

# check to see if we are using Raspberry Pi IO as an alarm
if args["alarm"] > 0:
    # import rasp IOs libs and set vars
    import RPi.GPIO as GPIO
    from time import sleep
    GPIO.setmode(GPIO.BCM)
    buzzer = 23
    GPIO.setup(buzzer, GPIO.OUT)
    print("[INFO] using Rasp IO alarm...")

EAR_THRESH = 0.3
EAR_CONSEC_FRAMES = 16
CONFIDENCE_FACE_DNN = 0.5  # minimum probability to filter weak detections
SUP_FACE = 0.7
DISTRACTION_THRESH = 1
DR_CONSEC_FRAMES = 16

MODEL_FILE = "res10_300x300_ssd_iter_140000.caffemodel"
PROTOTXT_FILE = "deploy.prototxt"

counter_ear = 0
counter_dr = 0
alarm_on = False
ear = 0
dr = 0

print("[INFO] loading models")
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')
net = cv.dnn.readNetFromCaffe(PROTOTXT_FILE, MODEL_FILE)

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

print("[INFO] starting video stream thread...")
cap = cv.VideoCapture(0)
time.sleep(2.0)

while True:
    begin = time.time()
    ret, frame = cap.read()
    orig = frame.copy()
    frame = cv.resize(frame, (400, 400))
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # detect faces in the grayscale frame
    rects = face_cascade.detectMultiScale(gray, scaleFactor=1.1,
                                          minNeighbors=5, minSize=(30, 30),
                                          flags=cv.CASCADE_SCALE_IMAGE)

    frontal_face_flag = False
    for (x, y, w, h) in rects:
        frontal_face_flag = True
        rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        cv.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 2)
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # EAR average
        ear = (leftEAR + rightEAR) / 2.0

        leftEyeHull = cv.convexHull(leftEye)
        rightEyeHull = cv.convexHull(rightEye)
        cv.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        if ear < EAR_THRESH:
            counter_ear += 1
            if counter_ear >= EAR_CONSEC_FRAMES:
                if not alarm_on:
                    alarm_on = True
                    if args["alarm"] > 0:
                        send_message("desatencao")
                        # Set buzzer and LEDs
                        GPIO.output(buzzer, GPIO.HIGH)
                        print("Beep")
                        None
                cv.putText(frame, "ALERTA DE FADIGA!", (10, 30),
                           cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            counter_ear = 0
            alarm_on = False
            if args["alarm"] > 0:
                GPIO.output(buzzer, GPIO.LOW)
                print("no Beep")
                # Reset buzzer and LEDs
                None

        cv.putText(frame, f"EAR: {np.round(ear, 2)}", (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        break

    if not frontal_face_flag:
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

            # text = "{:.2f}%".format(confidence * 100)
            # y = startY - 10 if startY - 10 > 10 else startY + 10
            # cv.putText(frame, text, (startX, y),
            #            cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

            cv.rectangle(frame, (startX, startY), (endX, endY),
                         (0, 0, 255), 2)

            face = frame[startY:startY+int((endY-startY)*SUP_FACE), startX:endX]
            face_gray = cv.cvtColor(face, cv.COLOR_BGR2GRAY)
            eyes = eye_cascade.detectMultiScale(face_gray, scaleFactor=1.1,
                                                minNeighbors=5, minSize=(30, 30),
                                                flags=cv.CASCADE_SCALE_IMAGE)

            eyes_pos_x = []
            for i, (ex, ey, ew, eh) in enumerate(eyes):
                if i >= 2:
                    break
                cv.rectangle(face, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                eyes_pos_x.append(ex + (ew/2))

            dr = distraction_ratio(eyes_pos_x)
            if dr is None or dr < DISTRACTION_THRESH:
                counter_dr += 1
                if counter_dr >= DR_CONSEC_FRAMES:
                    if not alarm_on:
                        alarm_on = True
                        if args["alarm"] > 0:
                            send_message("desatencao")
                            # Set buzzer and LEDs
                            GPIO.output(buzzer, GPIO.HIGH)
                            print("Beep")
                            None
                    cv.putText(frame, "ALERTA DE DSITRACAO!", (10, 30),
                               cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                counter_dr = 0
                alarm_on = False
                if args["alarm"] > 0:
                    GPIO.output(buzzer, GPIO.LOW)
                    print("no Beep")
                    # Reset buzzer and LEDs
                    None

        cv.putText(frame, f"DR: {dr}", (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Show the frame
    result = cv.resize(orig, (400, 400))
    cv.imshow("EyeTractor", cv.hconcat((frame, result)))
    key = cv.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

    print(f'loop time = {np.round(time.time() - begin, 2)}')

cv.destroyAllWindows()
