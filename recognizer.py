import os
import cv2
import numpy as np
import pandas as pd
from deepface import DeepFace
import json


class Recognize_verify:
    def __init__(self):
        path_to_json_file = "dataset/recorded_encodings/face_encodings.json"
        with open(path_to_json_file, "r+") as json_file:
            self.data = json.loads(json_file.read())

    def verify_faces(self):
        missing_list = []
        models = ['VGG-Face', 'Facenet', 'OpenFace', 'DeepFace', 'DeepID', 'Dlib']
        # for _ in self.data:
        # model = DeepFace.build(models)
            # use the data_pool variable to set the path for pre-recorded samples
        data_pool = "dataset/Pre_recorded_encodings"
        if DeepFace.find(img_path = np.array(self.data["encoding"]), db_path = data_pool, detector_backend = 'opencv')["verified"] == True:
            return self.data["passenger"], "has boarded the vehicle"
        else:
            missing_list.append(_.popitem()[0])

            
        for _ in missing_list:
            print(_)


obj = Recognize_verify()
obj.verify_faces()