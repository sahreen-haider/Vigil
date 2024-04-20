import cv2
import supervision as sv
from ultralytics import YOLO


def get_detections(source_path):
    image = cv2.imread(source_path)
    model = YOLO("yolov8s.pt")
    result = model(image)[0]
    detections = sv.Detections.from_ultralytics(result)
    return len(detections)


get_detections("PXL_20231104_072540596.PORTRAIT.jpg")