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
        encoding_data = dict()
        model_names = ["opencv", "ssd", "dlib", "mtcnn"]
        gen_path = "generated_data"
        # face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        cap = cv2.VideoCapture(0)
        # df = pd.read_csv("dataset/recorded_encodings/recorded_encode.csv")

        while True:
            cap.set(3, 640)     #set video width
            cap.set(4, 480)     #set video height
            ret, self.frame = cap.read()
            
            # using "SSD(single shot detector)" for its faster performance
            time_now = str(datetime.now()).split()
            try:
                self.detected_face = DeepFace.extract_faces(self.frame, detector_backend = model_names[1])
                faces = self.detected_face[0]["facial_area"]
                self.final_detected_face = self.detected_face[0]["face"]
                cv2.rectangle(self.frame, (faces["x"],faces["y"]), (faces["x"]+faces["w"], faces["y"]+faces["h"]), (255, 0, 0), 2)
                cv2.imshow("Detect Faces", self.frame)
                print("block 1 try executed")

            
            except Exception as E:
                cv2.imshow("Detect Faces", self.frame)
                print("block 1 except executed")

            
            if len(encoding_data) == 0: 
                encoding_data.update({f"candidate {counter}": self.final_detected_face.tolist()})
                cv2.imwrite("generated_data/candidate_{counter}.png", self.final_detected_face)
                counter += 1


            else:
                # print(encoding_data[f"candidate {counter-1}"])
                # if DeepFace.verify(np.array(encoding_data[list(encoding_data.keys())[-1]]), self.frame, model_name = "VGG-Face")["verified"] == True:
                if DeepFace.verify(gen_path+os.listdir("generated_data")[-1], self.frame, model_name = "VGG-Face")["verified"] == True:
                    print("same person")
                    continue


                else:
                    # encoding_data.update({f"candidate {counter}": self.final_detected_face})
                    var1 = self.final_detected_face
                    print('not same person rather a different person')
                    counter += 1    

            if cv2.waitKey(1) & 0xFF == ord("v"):
                break

        with open("dataset/recorded_encodings/face_encodings.json", "w") as json_file:
            json.dump(encoding_data, json_file, indent = 4)
        
        print("faces have been recorded")
        cap.release() 
        cv2.destroyAllWindows()

    
obj = Detect_verify()
obj.capture_live_faces()