import os
import numpy as np
import pandas as pd
import cv2
from deepface import DeepFace
from datetime import datetime
import json
from recognizer import *


class Detect_verify:
    def from_source(self):
        cap = cv2.VideoCapture("path to file")


    def capture_live(self):
        self.counter = 1
        self.encoding_data = []
        self.model_names = ["opencv", "ssd", "dlib", "mtcnn"]
        self.self.gen_path = "dataset/"
        # face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        cam = cv2.VideoCapture(0)

        while True:
            cam.set(3, 640)     #set video width
            cam.set(4, 480)     #set video height
            ret, _ = cam.read()
            
            # using "SSD(single shot detector)" for its faster performance

        cam.release() 
        cv2.destroyAllWindows()
    def process_frames(self):
        for _ in range(len(self.encoding_data)):
            try:
                self.detected_face = DeepFace.extract_faces(_, detector_backend = self.model_names[1])
                # faces = self.detected_face[0]["facial_area"]
                self.final_detected_face = self.detected_face[0]["facial_area"]
                # cv2.rectangle(_, (faces["x"],faces["y"]), (faces["x"]+faces["w"], faces["y"]+faces["h"]), (255, 0, 0), 2)
                cv2.imshow("Detect Faces", _)
            
            except Exception as E:
                cv2.imshow("Detect Faces", _)
                continue


            if len(os.listdir(self.gen_path+"recorded_encodings")) == 0: 
                np.save(self.gen_path+f"recorded_encodings/candidate_{counter}", _)
                counter += 1


            else:
                if DeepFace.verify(os.listdir(self.gen_path+"recorded_encodings")[-1], _, model_name = "VGG-Face")["verified"] == True:
                    continue


                else:
                    np.save(self.gen_path+f"recorded_encodings/candidate_{counter}", _)
                    counter += 1    

            if cv2.waitKey(1) & 0xFF == ord("c"):
                break

    
obj = Detect_verify()
obj.capture_live()



obj = Recognize_verify()
obj.verify_faces()