import os
import cv2
from deepface import DeepFace



cam = cv2.VideoCapture(0)

ret, frame = cam.read()

while True:
    counter = 0
    if len(os.listdir("dataset/recorded_data")) == 0:
        cv2.imshow("live feed", frame)
        file_path = "dataset/recorded_data"
        cv2.imwrite(file_path + "record" + str(counter) + ".jpg")

    else:
        recent_face = cv2.imread("dataset/recorded_data" + os.listdir("dataset/recorded_data")[-1])
        if Deepface.verify(recent_face, frame, model_name = "VGG-Face") == True:
            continue
        else:
            cv2.imshow("live feed", frame)
            file_path = "dataset/recorded_data"
            cv2.imwrite(file_path + "record" + str(counter) + ".jpg", frame)

