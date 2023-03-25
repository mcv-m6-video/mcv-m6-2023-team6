import itertools
import copy

def load_from_txt(path, threshold):
    """
    :param path: path file

    :return: list = [[frame,x1, y1, x2, y2, conf]]
    """
    detections = []
    with open(path) as f:
        lines = f.readlines()

    for l in lines:
        ll = l.split(",")
        if float(ll[6]) >= threshold:
            frame = int(ll[0]) - 1
            detections.append(
                [
                    frame,
                    float(ll[2]),
                    float(ll[3]),
                    float(ll[2]) + float(ll[4]),
                    float(ll[3]) + float(ll[5]),
                    float(ll[6]),
                ]
            )

    """Group the detected boxes by frame_id as a dictionary"""
    detections.sort(key=lambda x: x[0])
    detections = itertools.groupby(detections, key=lambda x: x[0])
    detections = {k: list(v) for k, v in detections}

    return detections

# INTERSECTION OVER UNION
def iou(box1, box2, threshold=0.9):
    if len(box1) > 4:
        box1 = box1[:4]
    """Return iou for a single a pair of boxes"""
    x11, y11, x12, y12 = box1
    x21, y21, x22, y22 = box2

    xA = max(x11, x21)
    yA = max(y11, y21)
    xB = min(x12, x22)
    yB = min(y12, y22)

    if xB < xA or yB < yA:
        interArea = 0
    else:
        interArea = max(xB - xA, 0) * max(yB - yA, 0)

    # respective area of ​​the two boxes
    box1Area = (x12 - x11) * (y12 - y11)
    box2Area = (x22 - x21) * (y22 - y21)
    

    # IOU
    iou_score = interArea / (box1Area + box2Area - interArea)


    return iou_score,iou_score >= threshold



def discard_overlaps(frame_boxes,threshold=0.9):
    discard = []
    boxes = copy.deepcopy(frame_boxes)
    for i in range(len(boxes)):
        boxA = [boxes[i][1],boxes[i][2],boxes[i][3],boxes[i][4]]
        for j in range(len(boxes)):
            boxB = [boxes[j][1],boxes[j][2],boxes[j][3],boxes[j][4]]
            if i == j:
                continue
            elif any(j in sublist for sublist in discard):
                continue
            else:
                if iou(boxA,boxB,threshold) == True:
                    discard.append([i,j])

    discard.sort(key=lambda x: x[1], reverse=True)
    for d in discard:
        del boxes[d[1]]

    return frame_boxes