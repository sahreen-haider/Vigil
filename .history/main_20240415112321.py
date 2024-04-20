import cv2
import supervision as sv
from ultralytics import YOLO


# def get_detections(source_path):
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    

    model = YOLO("yolov8s.pt")
    result = model(frame)[0]
    detections = sv.Detections.from_ultralytics(result)
    print(len(detections))

    cv2.imshow("live feed", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break    

    
cap.release()
cv2.destroyAllWindows()    