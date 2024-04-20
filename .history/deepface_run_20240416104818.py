import os
import numpy as numpy
import cv2
from deepface import DeepFace 


model_names = ["opencv", "ssd", "dlib", "mtcnn"]

def capture_live_faces():
    # face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # using "SSD(single shot detector)" for its faster performance
        detected_face = DeepFace.extract_faces(frames, detector_backend = "ssd")

        faces = detected_face[0]["facial_area"]
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow("Detect Faces", gray)


        if cv2.waitKey(1) & 0xFF == ord("v"):
            break


    cap.release()
    cv2.destroyAllWindows()    

capture_live_faces()