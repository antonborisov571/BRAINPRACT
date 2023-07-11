import cv2
import torch
from ultralytics import YOLO

model = YOLO('weights/yolov8.pt')
model.conf = 0.5

cap = cv2.VideoCapture(0)
cv2.namedWindow("YOLOv8 Inference", cv2.WINDOW_NORMAL)

while cap.isOpened():
    success, frame = cap.read()
    if success:
        results = model.predict(frame, conf=0.5)
        annotated_frame = results[0].plot()
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()