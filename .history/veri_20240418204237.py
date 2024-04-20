import os
import cv2
from deepface import DeepFace



rec = DeepFace.verify(img1_path = cv2.imread("dataset/Pre_recorded_data/content.jpg"), img2_path = "dataset/Pre_recorded_data/positive_sample.jpg", model_name = "VGG-Face")["verified"]
