import cv2

def calculate_iou(box1, box2):

    """
      Calculates Intersection over Union (IoU) between two bounding boxes.

      Args:
          box1: The first bounding box coordinates:
              [x1, y1, x2, y2] where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner.
          box2: The second bounding box coordinates (same format as box1).

      Returns:
          The IoU value between the two bounding boxes, a float between 0 and 1.
      """

    # Find the top-left corner of the overlapping rectangle
    x1 = max(box1[0], box2[0])  # Maximum of leftmost x coordinates
    y1 = max(box1[1], box2[1])  # Maximum of topmost y coordinates

    # Find the bottom-right corner of the overlapping rectangle
    x2 = min(box1[2], box2[2])  # Minimum of rightmost x coordinates
    y2 = min(box1[3], box2[3])  # Minimum of bottommost y coordinates

    # Calculate the area of the overlap
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    # Calculate the area of each individual bounding box
    box_1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box_2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou = inter_area / float(box_1_area + box_2_area - inter_area)
    return iou


def boxes_intersect(box1, box2):
    """
      Checks if two bounding boxes have any overlap.

      Args:
          box1: The first bounding box coordinates:
              [x1, y1, x2, y2] where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner.
          box2: The second bounding box coordinates (same format as box1).

      Returns:
          True if there is any overlap between the two bounding boxes, False otherwise.
      """
    if box2[0] > box1[2] or box2[2] < box1[0]:
        return False  # No overlap on X-axis
    if box2[1] > box1[3] or box2[3] < box1[1]:
        return False  # No overlap on Y-axis
    return True  # Overlap on both axes, hence intersection

def assign_hi_vis(person_boxes, high_vis_boxes):
    matches = []
    for hv_box in high_vis_boxes:
        best_iou = 0
        best_person = None
        for p_box in person_boxes:
            if boxes_intersect(hv_box, p_box):
                iou = calculate_iou(hv_box, p_box)
                if iou > best_iou:
                    best_iou = iou
                    best_person = p_box
        if best_person is not None:
            matches.append((hv_box, best_person, best_iou))
    return matches

def draw_matches(image, matches):
    """
      Visualizes the matches between high-visibility bounding boxes and person bounding boxes on an image.

      Args:
          image: A NumPy array representing the image on which to draw the matches.
          matches: A list of tuples containing information about each match.
              Each tuple consists of three elements:
                  - hv_box (list): A list of four elements representing the high-visibility bounding box coordinates:
                      [x1, y1, x2, y2] where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner.
                  - p_box (list): A list of four elements representing the person bounding box coordinates (same format as hv_box).
                  - iou (float): The Intersection over Union (IoU) value between the high-visibility and person bounding boxes.

      Returns:
          The modified image with bounding boxes and IoU labels drawn on it.
      """
    for hv_box, p_box, iou in matches:
        hv_color = (0, 255, 0)  # Define colour for hi-vis bbox
        p_color = (255, 0, 0)  # Define colour for person bbox

        # Draw high-vis box
        cv2.rectangle(image, (int(hv_box[0]), int(hv_box[1])), (int(hv_box[2]), int(hv_box[3])), hv_color, 4)
        cv2.putText(image, f'High-Vis IoU: {iou:.2f}', (int(hv_box[0]), int(hv_box[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, hv_color, 2)

        # Draw person box
        cv2.rectangle(image, (int(p_box[0]), int(p_box[1])), (int(p_box[2]), int(p_box[3])), p_color, 4)
        cv2.putText(image, 'Person', (int(p_box[0]), int(p_box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, p_color, 2)

    return image
