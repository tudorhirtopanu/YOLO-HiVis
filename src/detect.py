from ultralytics import YOLO


class YoloDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.person_class_id = 0
        self.high_vis_class_id = 1

    def detect_objects(self, source, confidence=0.5):

        person_boxes = []
        high_vis_boxes = []

        results = self.model(source=source, show=True, conf=confidence, save=True)

        for result in results:
            boxes = result.boxes.xyxy
            classes = result.boxes.cls

            for i in range(len(classes)):
                box = boxes[i]
                class_id = int(classes[i])

                if class_id == self.person_class_id:
                    person_boxes.append(box)
                elif class_id == self.high_vis_class_id:
                    high_vis_boxes.append(box)

            result.show()

        return person_boxes, high_vis_boxes
