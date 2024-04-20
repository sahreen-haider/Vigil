import os
import cv2
from deepface import DeepFace

# cam = cv2.VideoCapture(0)

# ret, frame = cam.read()


# print(DeepFace.verify("dataset/Pre_recorded_data/content.jpg", "dataset/Pre_recorded_data/negative sample2.jpg", model_name = "VGG-Face")["verified"])

print(os.list_dir("dataset/Pre_recorded_data")[-1])