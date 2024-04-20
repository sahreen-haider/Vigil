import os
import numpy as numpy
import cv2
from deepface.commons import functions

model_name = "VGG-Face"

def capture_live():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        return type(frame)

capture_live()            