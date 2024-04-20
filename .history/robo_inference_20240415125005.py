import os
from dotenv import load_dotenv, find_dotenv
import cv2
import supervision as supervision
from ultralytics import yolov8s


cap  = cvs2.VideoCapture(0)

while True:

    ret, frame = cap.read()


    model = get_roboflow_model(model_id="yolov8s-640", api_key = )