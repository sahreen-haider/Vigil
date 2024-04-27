import os
import cv2
from deepface import DeepFace


print(DeepFace.extract_faces("generated_data/Photo on 25-04-24 at 11.04 AM.jpg")[0]["confidence"]), print(DeepFace.extract_faces("generated_data/candidate_1.jpg")[0]["confidence"])

