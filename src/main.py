from detect import YoloDetector
from bbox_utils import draw_matches, assign_hi_vis
import cv2

MODEL_PATH = "/Users/tudor/PycharmProjects/YOLO-HiVis/models/HiVisModel.pt"
SOURCE = "/Users/tudor/PycharmProjects/YOLO-HiVis/images/example-img/ss7.png"
image = cv2.imread(SOURCE)
CONFIDENCE = 0.5

detector = YoloDetector(MODEL_PATH)
person_boxes, hi_vis_boxes = detector.detect_objects(SOURCE, CONFIDENCE)

matches = assign_hi_vis(person_boxes, hi_vis_boxes)
print(f'person boxes: {person_boxes}')
print(f'hi vis boxes: {hi_vis_boxes}')
print(f'matches: {matches}')

output_image = draw_matches(image, matches)

cv2.imshow('Output', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('output_with_matches.jpg', output_image)
