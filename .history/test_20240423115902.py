# import os
# import numpy as np
import pandas as pd
import json
import cv2
import numpy as np
from deepface import DeepFace

# data = pd.DataFrame(columns = ["no", 'binary'])

# data.loc[len(data)] = ["88", "09"]
# data.loc[len(data)] = ["99", "fg"]
# print(data)


# Create an initial dictionary
# dict_example = {}

# # Update the dictionary with new key-value pairs
# dict_example.update({'a': 1})
# dict_example.update({'b': 2})
# dict_example.update({'c': 3})

# # Print the updated dictionary
# print(dict_example)
# df = pd.read_csv('dataset/recorded_encodings/recorded_encode.csv')
# with open("dataset/recorded_encodings/face_encodings.json", "r") as f:
#     data = json.loads(f.read())
#     encoding = np.array(data["encoding"])

# print(type(encoding))
# cv2.imshow('live feed' ,encoding)
# data = {
#     "passenger": "zuhaib",
#     "encoding": [1, 2, 3],
#     "Date": "12/01/23",
#     "Time": "12:01:12"
# }
# df = df._append(data, ignore_index=True)
# cam = cv2.VideoCapture(0)
# detected_face = []
# 
# while True:
    # ret, frame = cam.read()
# 
    # try:
detected_face = DeepFace.extract_faces("generated_data/Photo on 17-04-24 at 6.10 PM.jpg", detector_backend = "ssd")
# cv2.imshow(detected_face[0])
print(detected_face)
    # except Exception as E:
        # cv2.imshow(frame)
    # 
    # if cv2.waitKey(1) & 0xFF == ord("q"):
        # break
# 
# cv2.imshow("Frame", detected_face)

    