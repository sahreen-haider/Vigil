import os
import numpy as numpy
import cv2
from deepface import DeepFace 


class Detect_verify:
    def capture_live_faces(self):
        model_names = ["opencv", "ssd", "dlib", "mtcnn"]
        # face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        cap = cv2.VideoCapture(0)
        while True:
            ret, self.frame = cap.read()
            self.gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

            # using "SSD(single shot detector)" for its faster performance
            try:
                self.detected_face = DeepFace.extract_faces(self.frame, detector_backend = "ssd")
                faces = self.detected_face[0]["facial_area"]
            
                cv2.rectangle(self.frame, (faces["x"],faces["y"]), (faces["x"]+faces["w"], faces["y"]+faces["h"]), (255, 0, 0), 2)

                cv2.imshow("Detect Faces", self.frame)
                break
            
            except:
                continue

            


            if cv2.waitKey(1) & 0xFF == ord("v"):
                break


        cap.release()
        cv2.destroyAllWindows()



    def verify_faces(self):
        models = ['VGG-Face', 'Facenet', 'OpenFace', 'DeepFace', 'DeepID', 'Dlib']
        # model = DeepFace.build(models)
        if self.detected_face:
            file_path = "generated_data/captured_sample.jpg"
            cv2.imwrite(file_path, self.frame)
            distance_score = DeepFace.find(img_path = file_path, db_path = "dataset" , detector_backend = "ssd")
            for _ in distance_score:
                print(type(_))


            
        return


obj = Detect_verify()
obj.capture_live_faces()
obj.verify_faces()