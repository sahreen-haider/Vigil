import os
import numpy as numpy
import cv2
import deepface.utils.commons

model_name = "VGG-Face"

def capture_live():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        print(type(frame))

        if cv2.waitKey(1) & 0xFF == ord(v):
            break

capture_live()            