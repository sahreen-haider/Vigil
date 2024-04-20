import numpy as np
import cv2


def capture_live_faces():
        counter = 0
        model_names = ["opencv", "ssd", "dlib", "mtcnn"]
        while True:
            # face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            cap = cv2.VideoCapture(0)
            cap.set(3, 640)     #set video width
            cap.set(4, 480)     #set video height
            ret, frame = cap.read()
            mirrored_frame = cv2.flip(frame, 1)
                            
            cv2.imshow("live feed", mirrored_frame)


            if cv2.waitKey(1) & 0xFF == ord("l"):
                  break
            



capture_live_faces()




        