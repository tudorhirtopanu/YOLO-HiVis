# YOLO-HiVis
<p float="left">
  <img src="https://github.com/tudorhirtopanu/YOLO-HiVis/assets/122214687/3cd1045c-c734-4d18-aba5-b35f04fb1f1d" width="45%" />
  <img src="https://github.com/tudorhirtopanu/YOLO-HiVis/assets/122214687/cf23c6a3-7ebb-409f-a709-f5b162654780" width="45%" />
</p>

**YOLO-HiVis** is an experimental computer vision project designed to detect and track individuals wearing high-visibility vests using a custom-trained YOLO model.

# Table of Contents

1. [Key Features](#key-features)
2. [Dependencies](#dependencies)
3. [Installation](#installation)
4. [Further Training the Model](#further-training-the-model)

## Key Features:

1. **Object Detection:** The project utilizes a custom-trained YOLO (You Only Look Once) model to perform object detection. It identifies individuals and high-visibility jackets within images or video frames.

2. **Intersection over Union (IoU) Algorithm:** After object detection, an Intersection over Union (IoU) algorithm is applied to handle overlapping bounding boxes of individuals and high-visibility jackets. This algorithm determines whether a person is wearing a high-visibility clothing by assessing the overlap between the detected objects.

3. **Deep SORT Algorithm:** Individuals identified as wearing high-visibility jackets are then passed into the Deep SORT (Simple Online and Realtime Tracking) algorithm for tracking. Deep SORT allows for the continuous monitoring of individuals wearing high-visibility gear across frames or video sequences.

**Note:** While the model demonstrates strong performance, there is room for improvement. The the project's dataset is available on [Kaggle](https://www.kaggle.com/datasets/tudorhirtopanu/yolo-highvis-and-person-detection-dataset), providing opportunities for further exploration and training.

## Dependencies

- OpenCV
- Ultralytics
- Tensorflow

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/tudorhirtopanu/YOLO-HiVis.git
   ```
2. Inside the project, install the deep sort repository as a dependency :
    ```bash
    git clone https://github.com/tudorhirtopanu/deep_sort.git
   ```
3. Install dependencies if you haven't already :
   ```bash
    pip install opencv-python ultralytics tensorflow
   ```
4. Run main.py

## Further Training the Model

1. In config.yaml specify the path to the dataset.

**Note:** You can download the dataset from [Kaggle](https://www.kaggle.com/datasets/tudorhirtopanu/yolo-highvis-and-person-detection-dataset)

2. Run ```Train.py``` specifying the ```model``` path.
