import os
import numpy as numpy
import pandas as pd
import cv2
from deepface import DeepFace
from datetime import datetime
import json


class Detect_verify:
    def capture_live_faces(self):
        counter = 0
        threshold = 10
        encoding_data = dict()
        model_names = ["opencv", "ssd", "dlib", "mtcnn"]
        # face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        cap = cv2.VideoCapture(0)
        df = pd.read_csv("dataset/recorded_encodings/recorded_encode.csv")

        while True:
            cap.set(3, 640)     #set video width
            cap.set(4, 480)     #set video height
            ret, self.frame = cap.read()
            # using "SSD(single shot detector)" for its faster performance
            try:
                self.detected_face = DeepFace.extract_faces(self.frame, detector_backend = "opencv")
                faces = self.detected_face[0]["facial_area"]
                cv2.rectangle(self.frame, (faces["x"],faces["y"]), (faces["x"]+faces["w"], faces["y"]+faces["h"]), (255, 0, 0), 2)
                cv2.imshow("Detect Faces", self.frame)
            
            except Exception as E:
                cv2.imshow("Detect Faces", self.frame)

            
            if len(df["title"]) == 0: 
                df = df._append([f"candidate {counter}", self.frame, str(datetime.now())])
                counter += 1

            else:
                try:
                    recent_face = df["encoding"].iloc[-1]
                    if DeepFace.verify(recent_face, self.frame, model_name = "VGG-Face")["verified"] == True:
                        continue

                    else:
                        df = df._append([f"candidate {counter}", self.frame, str(datetime.now())])
                        counter += 1
                except:
                    continue
            

            if cv2.waitKey(1) & 0xFF == ord("v"):
                break


        cap.release() 
        cv2.destroyAllWindows()
        df.to_csv("dataset/recorded_encodings/recorded_encode.csv")

        
obj = Detect_verify()
obj.capture_live_faces()


