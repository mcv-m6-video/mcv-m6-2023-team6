import itertools
import csv

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
    for i in range(len(frame_boxes)):
        boxA = [frame_boxes[i][1],frame_boxes[i][2],frame_boxes[i][3],frame_boxes[i][4]]
        for j in range(len(frame_boxes)):
            boxB = [frame_boxes[j][1],frame_boxes[j][2],frame_boxes[j][3],frame_boxes[j][4]]
            if i == j:
                continue
            elif any(j in sublist for sublist in discard):
                continue
            else:
                _,score = iou(boxA,boxB,threshold)
                if score == True:
                    discard.append([i,j])

    discard.sort(key=lambda x: x[1], reverse=True)
    for d in discard:
        del frame_boxes[d[1]]

    return frame_boxes


def write_to_csv_file(filename, data):
    # Open a new CSV file to write the tracker data
    with open(filename, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)

        # Write the header row to the CSV file
        writer.writerow(['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'x', 'y', 'z'])

        # Loop through each frame in the dictionary and write the object data to the CSV file
        for frame_number, object_list in data.items():
            for object_data in object_list:
                frame = object_data[0]
                left = object_data[1]
                top = object_data[2]
                width = object_data[3] - object_data[1]
                height = object_data[4] - object_data[2]
                conf = object_data[5]
                if len(object_data) < 7:
                    ob_id = -1
                else:
                    ob_id = object_data[6]
                # Write the object data to the CSV file
                writer.writerow([frame_number, ob_id, left, top, width, height, conf, -1, -1, -1])
