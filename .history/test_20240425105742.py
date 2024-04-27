import os
import cv2
from deepface import DeepFace


print(DeepFace.extract_faces("/Volumes/Extended/.Trashes/501/candidate_4.jpg"))
print(DeepFace.extract_faces("generated_data/candidate_1.jpg"))

