# YOLO-HiVis

YOLO-HiVis is a computer vision project that utilizes a custom trained YOLO model to detect and classify high-visibility vests and people in images or video streams. Alongside the ability to detect these objects, I also wanted to differentiate between the presence of a high-vis and someone wearing one. Furthermore, there exists the case where there is a high-vis being worn, but multiple person bounding boxes make it unclear as to who is wearing the high vis. My solution to this was to use IOU algorithms in order to ensure accurate identification even when multiple person bounding boxes overlap.

## Features

- Detection and classification of high-visibility vests & people in images or video streams.
- Algorithms for calculating intersection over union (IoU) and determining bounding box overlaps.
- Trained on over 25,000 labels in 8,000+ images
- Understand the state/context of high-vis, is it only present or is it being worn
- Identify who is wearing the high vis, even when there are overlapping bounding boxes for people

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
