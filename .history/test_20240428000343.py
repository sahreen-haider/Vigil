import os
import cv2
from deepface import DeepFace



    
# if DeepFace.extract_faces("generated_data/Photo on 17-04-24 at 6.10 PM 2.jpg", enforce_detection = False, detector_backend = "ssd")[0]["confidence"] > 0.9:
#     cv2.imwrite("generated_data/sample_1.jpg", DeepFace.extract_faces("generated_data/candidate_2.jpg", enforce_detection=False, detector_backend = "ssd")[0]["face"])
#     print("detected face")

# else:
#     print("face not detected or might be that high confidence is not achieved")



list_present = DeepFace.find("dataset/recorded_encodings/Photo on 17-04-24 at 9.43 AM.jpg", db_path = "generated_data", model_name = "VGG-Face")[0]["identity"]

print(len(list_present))