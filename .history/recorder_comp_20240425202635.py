import os
import numpy as np
import pandas as pd
import cv2
from deepface import DeepFace
from datetime import datetime
import json


class Detect_verify:
    def capture_live_faces(self):
        counter = 1
        model_names = ["opencv", "ssd", "dlib", "mtcnn", "VGG-Face"]
        # face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        cap = cv2.VideoCapture(0)
        encodings = dict()                 # use dict instead if it moves slow 
        # also change .update() to .update()
        while True:
            cap.set(3, 640)     #set video width
            cap.set(4, 480)     #set video height
            ret, self.frame = cap.read()
            # using "SSD(single shot detector)" for its faster performance
            try:
                self.detected_face = DeepFace.extract_faces(self.frame, detector_backend = model_names[1])
                faces = self.detected_face[0]["facial_area"]
                cv2.rectangle(self.frame, (faces["x"],faces["y"]), (faces["x"]+faces["w"], faces["y"]+faces["h"]), (255, 0, 0), 2)
                cv2.imshow("Detect Faces", self.frame)
            
            except Exception as E:
                cv2.imshow("Detect Faces", self.frame)
                continue

            
            if len(encodings) == 0: 
                encodings.update({f"candidate {counter}" : self.frame.tolist()})
                counter += 1

            else:

                if DeepFace.verify(np.array(encodings[f"candidate_{counter}"]), self.frame, model_name = model_names[-1])["verified"] == False:
                    encodings.update({f"candidate {counter}" : self.frame.tolist()})
                    print("not the same person")
                    counter += 1

                else:
                    print("it is the same person")
                    continue
            

            if len(encodings) == 3 or cv2.waitKey(1) & 0xFF == ord("q"):
                break
                
        file_path = "dataset/recorded_encodings/face_encodings.json"
        print("no of persons", len(encodings))
        with open(file_path, "w") as json_file:
            json.dump(encodings, json_file, indent = 4)
        cap.release() 
        cv2.destroyAllWindows()
        

        

obj = Detect_verify()
obj.capture_live_faces()