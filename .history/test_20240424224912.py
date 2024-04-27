import os
import cv2
path = "generated_data/"+os.listdir("generated_data")[-1]

image = cv2.imread(path) 

cv2.imshow("screen", image)


