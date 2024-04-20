import os
from deepface import DeepFace


print(DeepFace.verify("dataset/Pre_recorded_data/content.jpg", "generated_data/captured_sample.jpg", model_name = "VGG-Face"))