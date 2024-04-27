import os
import cv2
from deepface import DeepFace


mage_1 = DeepFace.extract_faces("/Volumes/Extended/.Trashes/501/candidate_4.jpg")
mage_2 = DeepFace.extract_faces("generated_data/candidate_1.jpg")

cv2.imshow('feed',mage_1[0]["face"])

cv2.waitKey(0)
cv2.destroyAllWindows()