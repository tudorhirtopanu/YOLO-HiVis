import cv2
from detect import YoloDetector
from bbox_utils import assign_hi_vis2

MODEL_PATH = "/Users/tudor/PycharmProjects/YOLO-HiVis/models/HiVisModel.pt"
yolo_detector = YoloDetector(model_path=MODEL_PATH)

# Open video capture for live camera feed
cap = cv2.VideoCapture(0)  # Use camera index 0 for default camera

while True:
    # Read frame from the video capture
    ret, frame = cap.read()
    if not ret:
        break  # Break the loop if there are no more frames

    person_boxes, high_vis_boxes = yolo_detector.detect_objects(frame, 0.6)
    # Draw bounding boxes around detected objects
    for box in person_boxes:
        # Convert box coordinates to integers
        box = tuple(map(int, box))
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.putText(frame, 'Person', (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    for box in high_vis_boxes:
        # Convert box coordinates to integers
        box = tuple(map(int, box))
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
        cv2.putText(frame, 'High Vis', (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    people_count = len(person_boxes)
    high_vis_count = len(high_vis_boxes)
    matches2, num_high_vis, num_people, num_people_with_hi_vis = assign_hi_vis2(person_boxes, high_vis_boxes)
    count_text = f"People: {people_count}, High Vis: {high_vis_count}, People wearing: {num_people_with_hi_vis}"
    cv2.putText(frame, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display the frame
    cv2.imshow('Live Feed', frame)

    # Check for key press to exit the loop
    key = cv2.waitKey(1)
    if key == ord('q'):  # Press 'q' to quit
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
