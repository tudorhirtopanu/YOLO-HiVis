# YOLO-HiVis

**YOLO-HiVis** is an experimental computer vision project designed to detect and track individuals wearing high-visibility vests using a custom-trained YOLO model.

## Key Features:

1. **Object Detection:** The project utilizes a custom-trained YOLO (You Only Look Once) model to perform object detection. It identifies individuals and high-visibility jackets within images or video frames.

2. **Intersection over Union (IoU) Algorithm:** After object detection, an Intersection over Union (IoU) algorithm is applied to handle overlapping bounding boxes of individuals and high-visibility jackets. This algorithm determines whether a person is wearing a high-visibility clothing by assessing the overlap between the detected objects.

3. **Deep SORT Algorithm:** Individuals identified as wearing high-visibility jackets are then passed into the Deep SORT (Simple Online and Realtime Tracking) algorithm for tracking. Deep SORT allows for the continuous monitoring of individuals wearing high-visibility gear across frames or video sequences.

**Note:** While the model demonstrates strong performance, there is room for improvement. The the project's dataset is available on [Kaggle](https://www.kaggle.com/datasets/tudorhirtopanu/yolo-highvis-and-person-detection-dataset), providing opportunities for further exploration and training.

## Dependencies

- Ultralytics
- OpenCV

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your_username/YOLO-HiVis.git
   ```
2. Install Dependencies:
    ```bash
    pip install ultralytics opencv-python
   ```
