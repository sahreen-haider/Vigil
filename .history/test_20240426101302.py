import os
import cv2
from deepface import DeepFace



    
print(DeepFace.extract_faces("generated_data/candidate_2.jpg", detector_backend = "VGG-Face"))