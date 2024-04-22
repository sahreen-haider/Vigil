import os
import numpy as numpy
import pandas as pd
import cv2
from deepface import DeepFace
from datetime import datetime
import json


class Detect_verify:
    global     encodings
    encodings ={}
    def capture_live_faces(self):
        counter = 1
        model_names = ["opencv", "ssd", "dlib", "mtcnn"]
        # face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        cap = cv2.VideoCapture(0)

        while True:
            cap.set(3, 640)     #set video width
            cap.set(4, 480)     #set video height
            ret, self.frame = cap.read()
            # using "SSD(single shot detector)" for its faster performance
            try:
                self.detected_face = DeepFace.extract_faces(self.frame, detector_backend = "ssd")
                faces = self.detected_face[0]["facial_area"]
                cv2.rectangle(self.frame, (faces["x"],faces["y"]), (faces["x"]+faces["w"], faces["y"]+faces["h"]), (255, 0, 0), 2)
                cv2.imshow("Detect Faces", self.frame)
            
            except Exception as E:
                cv2.imshow("Detect Faces", self.frame)

            
            if len(encodings) == 0: 
                encodings.update({f"passenger {counter}":self.frame})
                counter += 1

            else:
                try:
                    if DeepFace.verify(encodings.popitem()[1], self.frame, model_name = "VGG-Face")["verified"] == True:
                        print("it is the same person")
                        continue

                    else:
                        encodings.update({f"candidate {counter}": self.frame})
                        print("not the same person")
                        counter += 1
                except:
                    continue
            

            if len(encodings) == 3 or cv2.waitKey(1) & 0xFF == ord("q"):
                break
                
        file_path = "dataset/recorded_encodings/face_encodings.txt"
        print("no of persons", len(encodings))
        with open(file_path, "w") as text_file:
            text_file.write(str(encodings))
        cap.release() 
        cv2.destroyAllWindows()
        

        
obj = Detect_verify()
obj.capture_live_faces()
