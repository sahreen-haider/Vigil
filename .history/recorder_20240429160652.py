import os
import numpy as np
import pandas as pd
import cv2
from deepface import DeepFace
from datetime import datetime
# import json
# from recognizer import *


class Detect_verify:
    def __init__(self):
        self.model_names = ["opencv", "ssd", "Dlib", "mtcnn", "VGG-Face"]
        self.gen_path = "dataset/"

    def from_source(self, path_source):
        self.encoding_data = []
        cap = cv2.VideoCapture(path_source)

        if not cap.isOpened():
            print("not able to open the source")
            exit()
        
        while True:
            ret, frame = cap.read()

            if not ret:
                break
            
            cv2.imshow('feed', frame)

            self.encoding_data.append(frame)

            if cv2.waitKey(1) & 0xFF == ord("x"):
                break



    def capture_live(self):
        self.encoding_data = []
        # face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        cam = cv2.VideoCapture(0)

        while True:
            cam.set(3, 640)     #set video width
            cam.set(4, 480)     #set video height
            ret, frames = cam.read()
            self.encoding_data.append(frames)
            
            cv2.imshow("Live Feed", frames)
            # using "SSD(single shot detector)" for its faster performance
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cam.release() 
        cv2.destroyAllWindows()
    def process_frames(self):
        counter = 1
        for _ in self.encoding_data:
            try:
                self.detected_face = DeepFace.extract_faces(_, detector_backend = self.model_names[0])
                faces = self.detected_face[0]["facial_area"]
                self.final_detected_face = self.detected_face[0]["facial_area"]
                cv2.rectangle(_, (faces["x"],faces["y"]), (faces["x"]+faces["w"], faces["y"]+faces["h"]), (255, 0, 0), 2)
                # cv2.imshow("Detect Faces", _)
            
            except Exception as E:
                # cv2.imshow("Detect Faces", _)
                continue


            if len(os.listdir(self.gen_path+"recorded_encodings")) == 0: 
                np.save(self.gen_path+f"recorded_encodings/candidate_{counter}", _)
                counter += 1


            else:
                # file_loader = np.load(os.listdir(self.gen_path+"recorded_encodings")[-1])
                if DeepFace.verify(np.load(self.gen_path + os.listdir(self.gen_path+"recorded_encodings")[-1]), _, model_name = self.model_names[-1])["verified"] == True:
                    continue


                else:
                    np.save(self.gen_path+f"recorded_encodings/candidate_{counter}", _)
                    counter += 1    
    
obj = Detect_verify()
obj.capture_live()
obj.process_frames()



# obj = Recognize_verify()
# obj.verify_faces()