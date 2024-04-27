import os
import cv2
import numpy as np
import pandas as pd
from deepface import DeepFace
import json


class Recognize_verify:
    def verify_faces(self):
        missing_list = []
        not_registered = []
        models = ['VGG-Face', 'Facenet', 'OpenFace', 'DeepFace', 'DeepID', 'Dlib']
        # for _ in self.data:
        # model = DeepFace.build(models)
            # use the data_pool variable to set the path for pre-recorded samples
        data_pool = "dataset/Pre_recorded_encodings"
        for _ in range(len(os.listdir("dataset/recorded_encodings"))):
            enumerate_passengers = DeepFace.find(img_path = np.array(self.data["encoding"]), db_path = data_pool, model_name = "VGG-Face")[0]["identity"][0]
            if len(enumerate_passengers) >= 1:
                if enumerate_passengers.split(".")[0] in  os.listdir("dataset/Pre_recorded_encodings"):
                    continue
                else:
                    missing_list.append("enumerate_passengers")
            else:
                not_registered.append()
    
    return missing_list 

obj = Recognize_verify()
obj.verify_faces()