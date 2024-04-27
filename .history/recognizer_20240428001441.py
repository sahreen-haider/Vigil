import os
import cv2
import numpy as np
import pandas as pd
from deepface import DeepFace
import json


class Recognize_verify:
    def __init__(self):
        path_to_json_file = "generated_data/face_encodings.json"
        with open(path_to_json_file, "r+") as json_file:
            self.data = json.loads(json_file.read())

    def verify_faces(self):
        missing_list = []
        not_registered = []
        models = ['VGG-Face', 'Facenet', 'OpenFace', 'DeepFace', 'DeepID', 'Dlib']
        # for _ in self.data:
        # model = DeepFace.build(models)
            # use the data_pool variable to set the path for pre-recorded samples
        data_pool = "dataset/Pre_recorded_encodings"
        enumerate_passengers = DeepFace.find(img_path = np.array(self.data["encoding"]), db_path = data_pool, model_name = "VGG-Face")[0]["identity"][0]
        if len(enumerate_passengers) >= 1:
            return enumerate_passengers[0], "has boarded the vehicle"
        else:
            not_registered.append()

            
        for _ in missing_list:
            print(_)


obj = Recognize_verify()
obj.verify_faces()