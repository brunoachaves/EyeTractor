from imutils import face_utils
import numpy as np
import argparse
import time
import dlib
import cv2 as cv


# compute and return the euclidean distance between the two points
def euclidean_dist(pt_a, pt_b):
    return np.linalg.norm(pt_a - pt_b)


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


ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", required=True,
                help="path to where the face cascade resides")
ap.add_argument("-p", "--shape-predictor", required=True,
                help="path to facial landmark predictor")
ap.add_argument("-a", "--alarm", type=int, default=0,
                help="boolean used to indicate if Raspberry Pi IO should be used")
args = vars(ap.parse_args())

# check to see if we are using Raspberry Pi IO as an alarm
if args["alarm"] > 0:
    # import rasp IOs libs and set vars
    print("[INFO] using Rasp IO alarm...")

EAR_THRESH = 0.3
EAR_CONSEC_FRAMES = 16
counter = 0
alarm_on = False
ear = 0

print("[INFO] loading facial landmark predictor...")
detector = cv.CascadeClassifier(args["cascade"])
predictor = dlib.shape_predictor(args["shape_predictor"])

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

print("[INFO] starting video stream thread...")
cap = cv.VideoCapture(0)
time.sleep(1.0)

while True:
    ret, frame = cap.read()
    frame = cv.resize(frame, (450, 450))
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # detect faces in the grayscale frame
    rects = detector.detectMultiScale(gray, scaleFactor=1.1,
                                      minNeighbors=5, minSize=(30, 30),
                                      flags=cv.CASCADE_SCALE_IMAGE)

    # loop over the face detections
    for (x, y, w, h) in rects:
        rect = dlib.rectangle(int(x), int(y), int(x + w),
                              int(y + h))

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
            counter += 1
            if counter >= EAR_CONSEC_FRAMES:
                if not alarm_on:
                    alarm_on = True
                    if args["alarm"] > 0:
                        # Set buzzer and LEDs
                        None
                cv.putText(frame, "ALERTA DE FADIGA!", (10, 30),
                           cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # [ENVIA SINAL PARA A NUVEM]

        else:
            counter = 0
            alarm_on = False
            if args["alarm"] > 0:
                # Reset buzzer and LEDs
                None

    cv.putText(frame, f"EAR: {np.round(ear, 2)}", (300, 30),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # show the frame
    cv.imshow("Frame", frame)
    key = cv.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
# do a bit of cleanup
cv.destroyAllWindows()

