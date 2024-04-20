import os
from deepface import DeepFace


print(DeepFace.verify("dataset/Pre_recorded_data/content.jpg", "dataset/Pre_recorded_data/positive_sample.jpg", model_name = "VGG-Face"))