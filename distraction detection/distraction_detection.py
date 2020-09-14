from imutils import face_utils
import numpy as np
import time
import dlib
import cv2 as cv


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


EAR_THRESH = 0.3
EAR_CONSEC_FRAMES = 16
counter = 0
alarm_on = False
ear = 0
cascade_face = 'haarcascade_frontalface_default.xml'
cascade_ear = ' haarcascade_mcs_rightear.xml'
landmarks_file = 'shape_predictor_68_face_landmarks.dat'

print("[INFO] loading facial landmark predictor...")
ear_detector = cv.CascadeClassifier(cascade_ear)
face_detector = cv.CascadeClassifier(cascade_face)
landmarks_predictor = dlib.shape_predictor(landmarks_file)

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(jaw_start, jaw_end) = face_utils.FACIAL_LANDMARKS_IDXS["jaw"]

print("[INFO] starting video stream thread...")
cap = cv.VideoCapture(0)
time.sleep(1.0)

while True:
    ret, frame = cap.read()
    frame = cv.resize(frame, (500, 500))
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # detect faces in the grayscale frame
    # rects = ear_detector.detectMultiScale(gray, scaleFactor=1.1,
    #                                   minNeighbors=5, minSize=(30, 30),
    #                                   flags=cv.CASCADE_SCALE_IMAGE)
    rects = ear_detector.detectMultiScale(gray, 1.3, 5)

    # loop over the face detections
    for (x, y, w, h) in rects:

        cv.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), palette['blue'])
        dlib_rect = dlib.rectangle(int(x), int(y), int(x + w),
                                   int(y + h))

        # print(dlib_rect)
        shape = landmarks_predictor(gray, dlib_rect)
        shape = face_utils.shape_to_np(shape)

        jaw = shape[jaw_start:jaw_end]

        print(jaw.shape)
        if len(jaw) == 0:
            print('distraction alert!!!!!!!!!!!!!!!')

        jaw_hull = cv.convexHull(jaw)
        cv.drawContours(frame, jaw_hull, -1, palette['green'], 2)

        '''
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
    
        # EAR average
        ear = (leftEAR + rightEAR) / 2.0
    
        leftEyeHull = cv.convexHull(leftEye)
        rightEyeHull = cv.convexHull(rightEye)
        jaw_hull = cv.convexHull(jaw)
        cv.drawContours(frame, [leftEyeHull], -1, palette['red'], 2)
        cv.drawContours(frame, [rightEyeHull], -1, palette['red'], 2)
        cv.drawContours(frame, jaw_hull, -1, palette['green'], 2)
        
        if ear < EAR_THRESH:
            counter += 1
            if counter >= EAR_CONSEC_FRAMES:
                if not alarm_on:
                    alarm_on = True
    
                cv.putText(frame, "ALERTA DE FADIGA!", (10, 30),
                           cv.FONT_HERSHEY_SIMPLEX, 0.7, palette['red'], 2)
                # [ENVIA SINAL PARA A NUVEM]
    
        else:
            counter = 0
            alarm_on = False
        '''
    cv.putText(frame, f"EAR: {np.round(ear, 2)}", (300, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, palette['red'], 3)

    # show the frame
    cv.imshow("Frame", frame)
    key = cv.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

cv.destroyAllWindows()
