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
                        present_list.append(_)
                    else:
                        not_registered.append(_)

                # print(list(set(os.listdir("dataset/Pre_recorded_encodings")) - set("dataset/recorded_encodings")))

                    print("mising persons: ", [_ for _ in os.listdir("dataset/Pre_recorded_encodings") if _ not in os.listdir("dataset/recorded_encodings")])        
                except:
                    continue   
            else:
                continue        
        print("people who are present: ", present_list, "not registered:",not_registered)

# if __name__ != '__main__':
#     obj = Recognize_verify()
#     obj.verify_faces()