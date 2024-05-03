import os
import cv2
import numpy as np
import pandas as pd
from deepface import DeepFace
import json
class Recognize_verify:
    def verify_faces(self):
        present_list = []
        not_registered = []
        missing_list = []
        models = ['VGG-Face', 'Facenet', 'OpenFace', 'DeepFace', 'DeepID', 'Dlib']
        # for _ in self.data:
        # model = DeepFace.build(models)
            # use the data_pool variable to set the path for pre-recorded samples
        data_pool = "dataset/Pre_recorded_encodings/"
        for _ in os.listdir("dataset/recorded_encodings"):
            if ".jpg" in _:
                try:
                    enumerate_passengers = DeepFace.find(img_path = f"dataset/recorded_encodings/{_}", db_path = data_pool, model_name = 'Facenet512')[0]["identity"].to_list()
                    # enum = enumerate_passengers[0].split(".")[0]
                    if len(enumerate_passengers) >= 1:
                        passenger_name = enumerate_passengers[0].split('/')[-1].split('.')[0]
                        present_list.append(passenger_name)
                    else:
                        not_registered.append(_)
                # print(list(set(os.listdir("dataset/Pre_recorded_encodings")) - set("dataset/recorded_encodings")))
                    missing_list = [x for x in os.listdir("dataset/Pre_recorded_encodings") if x not in os.listdir("dataset/recorded_encodings")]
                except:
                    continue   
            else:
                continue        
        with open("generated_data/dump.txt", "w") as file:
            file.write(f"passengers present: {present_list}, passengers_missing: {missing_list}, passengers_not_registered: {not_registered}")
        # return present_list, not_registered, missing_list