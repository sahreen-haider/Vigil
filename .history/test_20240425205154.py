import os
import cv2
from deepface import DeepFace

cam = cv2.VideoCapture(0)
while True:
ret, frame = cam.read()

DeepFace.find("", db_path="generated_data")