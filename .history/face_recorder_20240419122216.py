import os
import numpy as numpy
import pandas as pd
import cv2
from deepface import DeepFace 


class Detect_verify:
    def capture_live_faces(self):
        counter = 0
        model_names = ["opencv", "ssd", "dlib", "mtcnn"]
        # face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        cap = cv2.VideoCapture(0)
        cap.set(3, 640)     #set video width
        cap.set(4, 480)     #set video height
        ret, self.frame = cap.read()
                        
        # using "SSD(single shot detector)" for its faster performance
        self.detected_face = DeepFace.extract_faces(self.frame, detector_backend = "ssd")
        faces = self.detected_face[0]["facial_area"]

        while True:
            df = pd.read_csv("dataset/recorded_encodings/recorded_encode.csv")
            if len() != 0:
                cv2.rectangle(self.frame, (faces["x"],faces["y"]), (faces["x"]+faces["w"], faces["y"]+faces["h"]), (255, 0, 0), 2)
                cv2.imshow("Detect Faces", self.frame)
                file_path = "/Volumes/Extended/projects/Vigil/dataset/recorded_data/recorded_sample"+str(counter)+".jpg"
                cv2.imwrite(file_path, self.frame)
                counter += 1

            # except:
                # self.gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            else:
                try:
                    recent_path = "dataset/recorded_data"
                    recent_face = cv2.imread(recent_path + os.listdir("dataset/recorded_data")[1])
                    if DeepFace.verify(recent_face, self.frame, model_name = "VGG-Face")["verified"] == True:
                        continue

                    else:
                        cv2.rectangle(self.frame, (faces["x"],faces["y"]), (faces["x"]+faces["w"], faces["y"]+faces["h"]), (255, 0, 0), 2)
                        cv2.imshow("Detect Faces", self.frame)
                        file_path = "/Volumes/Extended/projects/Vigil/dataset/recorded_data/recorded_sample"+str(counter)+".jpg"
                        cv2.imwrite(file_path, self.frame)
                        counter += 1

            

            if cv2.waitKey(1) & 0xFF == ord("v"):
                break


        cap.release() 
        cv2.destroyAllWindows()

        
obj = Detect_verify()
obj.capture_live_faces()



