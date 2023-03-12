import numpy as np
from collections import defaultdict

# Intersection over Union (IoU)
def iou(box1, box2):
    x11, y11, x12, y12 = box1
    x21, y21, x22, y22 = box2
    xA = np.max(x11, x21)
    yA = np.max(y11, y21)
    xB = np.min(x12, x22)
    yB = np.min(y12, y22)
    interArea = np.max(0, xB - xA + 1) * np.max(0, yB - yA + 1)
    boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
    boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
    iou = interArea / np.float64(boxAArea + boxBArea - interArea)
    return iou


# Generate noisy boxes for testing
def generate_noisy_boxes(gt_boxes, del_prob, mean, std):
    """
    :gt_boxes: ground truth bounding boxes dict
    :del_prob: probability to delete bounding boxes
    :return: dictionary with the noisy bounding boxes
    """
    noisy_bboxes = defaultdict(list)
    for frame,bboxes in gt_boxes.items():
        for bbox in bboxes:
            if np.random.random() > del_prob:
                xtl, ytl, xbr, ybr = bbox
                noise = np.random.normal(mean,std,4)
                noisy_bboxes[frame].append([xtl+noise[0], ytl+noise[1], xbr+noise[2], ybr+noise[3]])

    return noisy_bboxes

# Average Precision (AP) for Object Detection
def ap():
    pass
