import cv2
import torch

model = torch.hub.load("ultralytics/yolov5", 'custom', path='weights/yolov5.pt')
model.conf = 0.5

cap = cv2.VideoCapture(0)
cv2.namedWindow("YOLOv5 Inference", cv2.WINDOW_NORMAL)

while True:
    success, frame = cap.read()
    if success:
        results = model(frame)
        results.render()
        cv2.imshow("YOLOv5 Inference", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()