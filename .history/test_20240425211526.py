import os
import cv2
from deepface import DeepFace

cam = cv2.VideoCapture(0)
while True:
    ret, frame = cam.read()
    try:
        faces = DeepFace.extract_faces(frame, detector_backend="ssd")[0]["face"]
        cv2.imshow("feed", frame)
    except Exception as E:
        cv2.imshow("feed", frame)
        continue

    # if len(DeepFace.find(frame, db_path="generated_data", model_name="VGG-Face")) != 0:
    #     print("the person's encoding is already present")
    
    # else:
    #     print("the person is appearing for the first time")
    with open("generated_data/sample_1.npy", "w") as np_file:
        np_file.write(frame)
    
    print(type(DeepFace.find(frame, db_path="generated_data", model_name="VGG-Face")[0]))
    
    break
    
cv2.waitKey(1)
cv2.destroyAllWindows()