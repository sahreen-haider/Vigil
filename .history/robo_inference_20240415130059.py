import os
from dotenv import load_dotenv, find_dotenv
import cv2
import supervision as supervision
from ultralytics import yolov8s


load_dotenv(find_dotenv())

cap  = cvs2.VideoCapture(0)

while True:

    ret, frame = cap.read()


    model = get_roboflow_model(model_id="yolov8s-640", api_key = os.getenv("roboloflow_api_key"))

    result = model.infer(frame)[0]
    directions = sv.Detections.from_inference(result)
    
    print(len(detections))

    cv2.imshow("live feed", frame)

    if cv2.waitKey(1) & 0xFF == ord("x"):
        break


cap.release()
cv2.destroyAllWindows()    
