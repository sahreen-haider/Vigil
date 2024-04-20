import os
import cv2
import numpy as np
import pandas as pd
from deepface import DeepFace


class Recognise_verify:
    def __init__(self):
        df = pd.read_csv("/Volumes/Extended/projects/Vigil/dataset/recorded_encodings/recorded_encode.csv")
        df_comp = pd.read_csv("place_holder")


    def verify_faces(self):
        models = ['VGG-Face', 'Facenet', 'OpenFace', 'DeepFace', 'DeepID', 'Dlib']
        for _ in range(len(df)):
        # model = DeepFace.build(models)
            if self.detected_face:
                distance_score = DeepFace.find(img_path = file_path, db_path = "dataset" , detector_backend = "ssd")
                print(distance_score[0]["identity"])
            
        return


obj = Detect_verify()
obj.capture_live_faces()
obj.record_faces()
# obj.verify_faces()
