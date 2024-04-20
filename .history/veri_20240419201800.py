import numpy as np
import pandas as pd
import datetime
import cv2
from deepface import DeepFace

# df = pd.read_csv("dataset/recorded_encodings/recorded_encode.csv")
# df[["title", "encoding", "timestamp"]] = [["fgjfg", "dfgg", "345"]]

# print(df["encoding"].iloc[-1])



def capture_live_faces():
        counter = 0
        model_names = ["opencv", "ssd", "dlib", "mtcnn"]
        while True:
            # face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            cap = cv2.VideoCapture(0)
            cap.set(3, 640)     #set video width
            cap.set(4, 480)     #set video height
            ret, self.frame = cap.read()
                            
            # using "SSD(single shot detector)" for its faster performance
            self.detected_face = DeepFace.extract_faces(self.frame, detector_backend = "ssd")
            faces = self.detected_face[0]["facial_area"]
            cv2.rectangle(self.frame, (faces["x"],faces["y"]), (faces["x"]+faces["w"], faces["y"]+faces["h"]), (255, 0, 0), 2)
            cv2.imshow("Detect Faces", self.frame)
            

            if cv2.waitKey(1) & 0xFF == ord("c"):
                  break
            

capture_live_faces()



        