import os
import cv2
from deepface import DeepFace



cam = cv2.VideoCapture(0)

ret, frame = cam.read()
cam.set(3, 640)     #set video width
cam.set(4, 480)     #set video height
while True:
    counter = 0
    file_path = "dataset/recorded_data/" + "record_" + str(counter) + ".jpg"
    if len(os.listdir("dataset/recorded_data")) == 0:
        cv2.imshow("live feed", frame)
        cv2.imwrite(file_path, frame)
        counter += 1

    else:
        recent_face = cv2.imread("dataset/recorded_data/" + os.listdir("dataset/recorded_data")[-1])
        if DeepFace.verify(recent_face, frame, model_name = "VGG-Face") == True:
            continue
        else:
            
            cv2.imwrite(file_path, frame)
            cv2.imshow("live feed", frame)
            counter += 1

