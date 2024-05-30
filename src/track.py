import os
import cv2
from ultralytics import YOLO

video_path = '/Users/tudor/Desktop/people.mp4'

cap = cv2.VideoCapture(video_path)

ret, frame = cap.read()

model = YOLO('/Users/tudor/PycharmProjects/YOLO-HiVis/models/HiVisModel.pt')

while ret:

    results = model(frame)

    for result in results:
        detections = []
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r

    cv2.imshow('frame', frame)
    cv2.waitKey(25)

    ret, frame = cap.read()

cap.release()
cv2.destroyAllWindows()
