import os
import cv2
from deepface import DeepFace



    
print(type(DeepFace.find(frame, db_path="generated_data", model_name="VGG-Face")[0]))