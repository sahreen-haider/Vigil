import os
import cv2
from deepface import DeepFace

cam = cv2.VideoCapture(0)
while True:
    ret, frame = cam.read()
    try:
        faces = DeepFace.extract_faces(frame, detector_backend="ssd")[0]["face"]
        cv2.imshow("feed", frame)
    except Exception as E:
        cv2.imshow("feed", frame)
        continue


    print(DeepFace.find(faces, db_path="generated_data", model_name="VGG-Face"))