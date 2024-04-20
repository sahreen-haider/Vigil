import os
import numpy as numpy
import pandas as pd
import cv2
from deepface import DeepFace
from datetime import datetime


class Detect_verify:
    def capture_live_faces(self):
        counter = 0
        encoding_prior = []
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

            
            if len(encoding_prior) == 0:
                print(f"encodings_recorded {counter}")
                encoding_prior = self.frame
                counter += 1


            else:
                try:
                    if DeepFace.verify(encoding_prior, self.frame, model_name = "VGG-Face")["verified"] == True:
                        continue

                    else:
                        print(f"encodings_recorded {counter}")
                        counter += 1
                except:
                    continue
            

            if cv2.waitKey(1) & 0xFF == ord("v"):
                break

        # cap.release() 
        # cv2.destroyAllWindows()

        
obj = Detect_verify()
obj.capture_live_faces()