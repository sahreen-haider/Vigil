import os
import cv2
import numpy as np
from deepface import DeepFace



    
# if DeepFace.extract_faces("generated_data/Photo on 17-04-24 at 6.10 PM 2.jpg", enforce_detection = False, detector_backend = "ssd")[0]["confidence"] > 0.9:
#     cv2.imwrite("generated_data/sample_1.jpg", DeepFace.extract_faces("generated_data/candidate_2.jpg", enforce_detection=False, detector_backend = "ssd")[0]["face"])
#     print("detected face")

# else:
#     print("face not detected or might be that high confidence is not achieved")



# list_present = DeepFace.find("dataset/recorded_encodings/Photo on 17-04-24 at 9.43 AM.jpg", db_path = "generated_data", model_name = "VGG-Face")[0]["identity"]

# print(len(list))

# cv2.imshow("image", np.load("dataset/recorded_encodings/candidate_1.npy"))
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# print(DeepFace.verify(np.load("dataset/recorded_encodings/candidate_1.npy"), "generated_data/Photo on 17-04-24 at 6.10 PM 3.jpg", model_name = "VGG-Face", detector_backend="opencv"))

im = np.load('/Users/zuhaib/Code/iQ/vigil/dataset/recorded_encodings/candidate_5.npy')
cv2.imshow('Image', im)
cv2.waitKey(0)
cv2.destroyAllWindows()

# a = np.load('dataset/recorded_encodings/candidate_4.npy').flatten()
# b = np.load('dataset/recorded_encodings/candidate_5.npy').flatten()

# dot_product = np.dot(a, b)
# magnitude_A = np.linalg.norm(a)
# magnitude_B = np.linalg.norm(b)

# cosine_similarity = dot_product / (magnitude_A * magnitude_B)
# print(f"Cosine Similarity using NumPy: {np.round(cosine_similarity, decimals=1)}")

# print(a.shape)