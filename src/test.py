import os
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

image_path = "/Users/tudor/PycharmProjects/YOLO-HiVis/src/img3.jpg"  # Replace 'your_image.jpg' with your actual image filename
image_path_out = '{}_out.jpg'.format(os.path.splitext(image_path)[0])

model = YOLO("/Users/tudor/PycharmProjects/YOLO-HiVis/runs/detect/train2/weights/last.pt")  # load a custom model

# Load the image
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for displaying with matplotlib

# Run inference
results = model(image)[0]

# Threshold for filtering detections
threshold = 0.5

# Process results and draw bounding boxes
for result in results.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = result

    if score > threshold:
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(image, results.names[int(class_id)].upper(), (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

# Save the output image
cv2.imwrite(image_path_out, image)

# Display the image with bounding boxes using matplotlib
plt.figure(figsize=(10, 10))
plt.imshow(image_rgb)
plt.axis('off')
plt.show()
