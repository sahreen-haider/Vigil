import os
import cv2
from deepface import DeepFace


def capture_live():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        cv2.imshow("live feed", frame)

        if cv2.waitKey(1) & 0xFF == ord('x'):
            break