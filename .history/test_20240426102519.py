import os
import cv2
from deepface import DeepFace



    
if DeepFace.extract_faces("generated_data/candidate_2.jpg", detector_backend = "ssd")[0]["confidence"] > 0.9:
    print(DeepFace.extract_faces("generated_data/candidate_2.jpg", detector_backend = "ssd")[0]["facial_area"])
    print("detected face")

else:
    print("face not detected or might be that high confidence is not achieved")
    