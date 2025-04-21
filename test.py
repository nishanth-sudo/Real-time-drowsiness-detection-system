import os
import cv2
import dlib
import time
import imutils
import argparse
import numpy as np
from threading import Thread, Lock
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import pygame

# Thread-safety
alarm_lock = Lock()

pygame.mixer.init()

# Global flags
alarm_status = False
alarm_status2 = False
saying = False

def sound_alarm(path):
    global alarm_status
    global alarm_status2
    global saying

    if os.path.exists(path):
        with alarm_lock:
            if alarm_status and not pygame.mixer.music.get_busy():
                print("[ALERT] Drowsiness alarm sounding...")
                pygame.mixer.music.load(path)
                pygame.mixer.music.play()
            elif alarm_status2 and not saying:
                print("[ALERT] Yawn alarm sounding...")
                saying = True
                pygame.mixer.music.load(path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)
                saying = False
    else:
        print(f"[ERROR] Alarm file not found: {path}")

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def final_ear(shape):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)
    return ((leftEAR + rightEAR) / 2.0, leftEye, rightEye)

def lip_distance(shape):
    top_lip = np.concatenate((shape[50:53], shape[61:64]))
    low_lip = np.concatenate((shape[56:59], shape[65:68]))
    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)
    return abs(top_mean[1] - low_mean[1])

# ------------------------------
# Argument parser
# ------------------------------
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--webcam", type=int, default=0, help="index of webcam on system")
ap.add_argument("-a", "--alarm", type=str, default="Vaigasi-Nilavae (mp3cut.net).mp3", help="path alarm .mp3 file")
args = vars(ap.parse_args())

EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 30
YAWN_THRESH = 20
COUNTER = 0

# ------------------------------
# Initialization
# ------------------------------
print("-> Loading the predictor and detector...")
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

print("-> Starting Video Stream")
vs = VideoStream(src=args["webcam"]).start()
time.sleep(1.0)

# ------------------------------
# Main Loop
# ------------------------------
try:
    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detector.detectMultiScale(gray, scaleFactor=1.1,
                                          minNeighbors=5, minSize=(30, 30),
                                          flags=cv2.CASCADE_SCALE_IMAGE)

        for (x, y, w, h) in faces:
            rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            ear, leftEye, rightEye = final_ear(shape)
            lip_dist = lip_distance(shape)

            # Draw eye & lip contours
            cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [shape[48:60]], -1, (0, 255, 0), 1)

            # Drowsiness detection logic (3 zones)
            if ear < 0.20:
                COUNTER += 1
                if COUNTER >= EYE_AR_CONSEC_FRAMES and not alarm_status:
                    alarm_status = True
                    t = Thread(target=sound_alarm, args=(args["alarm"],))
                    t.daemon = True
                    t.start()
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            elif ear < 0.25:
                cv2.putText(frame, "Eyes getting drowsy...", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                COUNTER = 0
                alarm_status = False
                pygame.mixer.music.stop()

            else:
                COUNTER = 0
                if alarm_status:
                    pygame.mixer.music.stop()
                alarm_status = False

            # Yawn detection
            if lip_dist > YAWN_THRESH:
                cv2.putText(frame, "YAWN ALERT!", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if not alarm_status2 and not saying:
                    alarm_status2 = True
                    t = Thread(target=sound_alarm, args=(args["alarm"],))
                    t.daemon = True
                    t.start()
            else:
                if alarm_status2:
                    pygame.mixer.music.stop()
                alarm_status2 = False

            # Always show metrics
            cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"YAWN: {lip_dist:.2f}", (300, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if len(faces) == 0:
            cv2.putText(frame, "No face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
finally:
    vs.stream.release()
    cv2.destroyAllWindows()
    pygame.mixer.music.stop()
    pygame.mixer.quit()
    print("-> Video stream stopped and window closed.")
