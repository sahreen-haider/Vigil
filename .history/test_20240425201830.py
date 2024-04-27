import os
import cv2
from deepface import DeepFace


# mage_1 = DeepFace.extract_faces("generated_data/Photo on 25-04-24 at 11.04 AM.jpg", enforce_detection = False)
# mage_2 = DeepFace.extract_faces("generated_data/candidate_1.jpg")

# cv2.imshow('feed',mage_1[0]["face"])

# cv2.waitKey(0)
# cv2.destroyAllWindows()


dicto = {1:"one", 2:"two", 3:"three"}

print(dicto.keys()[-1])