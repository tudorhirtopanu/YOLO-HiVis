import cv2
from ultralytics import YOLO
from tracker import Tracker
import random
from bbox_utils import calculate_iou

colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]

def track_people_in_hi_vis(model_path, confidence_threshold=0.4, iou_threshold=0.3):
    """
        Detects and tracks people wearing high-visibility jackets using YOLO model.

        Args:
            model_path (str): Path to YOLO model.
            confidence_threshold (float, optional): Confidence threshold for object detection. Defaults to 0.4.
            iou_threshold (float, optional): Intersection over Union threshold for filtering detections. Defaults to 0.3.
        """

    # Initialise YOLO model and tracker for object tracking
    model = YOLO(model_path)
    tracker = Tracker()

    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to open camera")
        return

    # Loop to capture frames from camera
    while True:

        # Read frame from camera
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break

        # Perform object detection using YOLO model on the frame
        results = model(frame)

        # Initialize lists to store detected persons and high visibility jackets
        persons = []
        high_vis = []

        # Iterate through the detection results
        for result in results:
            # Iterate through each detected object
            for r in result.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = r
                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)
                score = float(score)
                class_id = int(class_id)

                # Filter objects based on confidence threshold
                if score > confidence_threshold:
                    # Check if detected object is a person or high visibility jacket
                    if class_id == 0:
                        persons.append([x1, y1, x2, y2, score])
                    elif class_id == 1:
                        high_vis.append([x1, y1, x2, y2, score])

            # Initialise list to store filtered persons (person wearing high vis jacket)
            filtered_persons = []

            # Iterate through detected persons
            for person in persons:
                person_bbox = person[:4]

                # Iterate through detected high vis jackets
                for jacket in high_vis:
                    jacket_bbox = jacket[:4]

                    # Calculate IoU between person and jacket bbox
                    iou = calculate_iou(person_bbox, jacket_bbox)

                    # If IoU is above threshold, consider person as wearing high vis
                    if iou > iou_threshold:
                        filtered_persons.append(person)
                        break

            # Update tracker with filtered persons for object tracking
            tracker.update(frame, filtered_persons)

            # Draw bounding boxes and track IDs on the frame
            for track in tracker.tracks:
                bbox = track.bbox
                x1, y1, x2, y2 = bbox
                track_id = track.track_id
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), colors[track_id % len(colors)], 3)

        cv2.imshow('frame', frame)
        cv2.waitKey(25)

    cap.release()
    cv2.destroyAllWindows()
