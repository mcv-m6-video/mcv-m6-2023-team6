import numpy as np
from collections import defaultdict
import random
import copy

# Intersection over Union (IoU)
def iou(box1, box2):
    if len(box1) > 4:
        box1=box1[:4]
    x11, y11, x12, y12 = box1
    x21, y21, x22, y22 = box2
    xA = np.maximum(x11, x21)
    yA = np.maximum(y11, y21)
    xB = np.minimum(x12, x22)
    yB = np.minimum(y12, y22)
    interArea = np.maximum(0, xB - xA + 1) * np.maximum(0, yB - yA + 1)
    boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
    boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
    iou = interArea / np.float64(boxAArea + boxBArea - interArea)
    return iou

# Generate noisy boxes for testing
def generate_noisy_boxes(gt_boxes, del_prob,gen_prob, mean, std,frame_shape=[1080, 1920]):
    """
    :gt_boxes: ground truth bounding boxes dict
    :del_prob: probability to delete bounding boxes
    :gen_prob: probability to generate bounding boxes
    :return: dictionary with the noisy bounding boxes
    """
    noisy_bboxes = []
    gt_total = 0
    for frame,bboxes in gt_boxes.items():
        for bbox in bboxes:
            gt_total += 1
            if np.random.random() > del_prob:
                xtl, ytl, xbr, ybr = bbox
                noise = np.random.normal(mean,std,4)
                noisy_bboxes.append([frame,xtl+noise[0], ytl+noise[1], xbr+noise[2], ybr+noise[3]])
                w = xbr - xtl
                h = ybr - ytl

        if np.random.random() <= gen_prob:
            x = np.random.randint(w, frame_shape[1]-w)
            y = np.random.randint(h, frame_shape[0]-h)
            noisy_bboxes.append([frame,x-w/2, y-w/2, x+w/2, y+w/2])


    return noisy_bboxes, gt_total

# Average Precision (AP) for Object Detection
def mean_AP_Pascal_VOC(gt_boxes,N_gt,predicted_boxes):
    """
    :gt_boxes: ground truth bounding boxes dict
    :N_gt: Total of ground truth bounding boxes
    :predicted_boxes: predicted bounding boxes
    :return: mean IOU, average precision
    """
    mIOU = 0
    tp = np.zeros(len(predicted_boxes))
    fp = np.zeros(len(predicted_boxes))
    gt_detected = copy.deepcopy(gt_boxes)

    for i in range(len(predicted_boxes)):
        frame = predicted_boxes[i][0]
        predicted = predicted_boxes[i][1:5]
        gt = gt_detected[frame]
        iou_score = []
        for b in range(len(gt)):
            iou_score.append(iou(gt[b],predicted))

        id = np.argmax(iou_score)
        max_iou = iou_score[id]
        mIOU += max_iou

        if max_iou >= 0.5:
            if len(gt_detected[frame][id]) == 4:
                gt_detected[frame][id].append(True)
                tp[i] = 1
            else:
                fp[i] = 1
        else:
            fp[i] = 1

    tp = np.cumsum(tp)
    fp = np.cumsum(fp)

    recall = tp/ float(N_gt)
    precision = tp/np.maximum(tp + fp, np.finfo(np.float64).eps)

    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        if np.sum(recall >= t) == 0:
            p = 0
        else:
            p = np.max(precision[recall >= t])
        ap = ap + p / 11.0

    return mIOU/len(predicted_boxes), ap


def compute_confidences_ap(gt_boxes,N_gt,predicted_boxes,N=10):
    """ 
    Randomly generates the order of the bounding boxes to calculate the average precision (N times). 
    Average values will be returned.
    """
    ap_scores = []
    for i in range(N):
        random.shuffle(predicted_boxes)
        mIOU, ap = mean_AP_Pascal_VOC(gt_boxes,N_gt,predicted_boxes)
        ap_scores.append(ap)
        

    return sum(ap_scores)/len(ap_scores),mIOU
