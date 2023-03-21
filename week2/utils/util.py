import os
from collections import defaultdict

import cv2
import xmltodict
import numpy as np

""" def load_from_xml(path):

    frame_dict = defaultdict(list)
    for event, elem in ET.iterparse(path, events=('start',)):
        if elem.tag == 'track' and elem.attrib.get('label') == 'car':
            for x in (child.attrib for child in elem):
                frame = f"f_{x['frame']}"
                frame_dict[frame].append([float(x['xtl']), float(x['ytl']),
                                          float(x['xbr']), float(x['ybr'])])
    return frame_dict """


def load_from_xml(path):
    """

    :param path: path file

    :return: dict[frame_num] = [[x1, y1, x2, y2]]
    """

    with open(path) as f:
        tracks = xmltodict.parse(f.read())["annotations"]["track"]

    gt = defaultdict(list)
    num_iter = 0
    for track in tracks:
        label = track["@label"]
        boxes = track["box"]
        for box in boxes:
            if label == "car" and box['attribute']['#text'].lower() == 'false' :
                frame = int(box["@frame"])
                frame = f"f_{frame}"
                gt[frame].append(
                    [
                        float(box["@xtl"]),
                        float(box["@ytl"]),
                        float(box["@xbr"]),
                        float(box["@ybr"]),
                    ]
                )
                num_iter += 1

            else:
                continue

    return gt


def load_from_txt(path):
    """
    :param path: path file

    :return: list = [[frame,x1, y1, x2, y2, conf]]
    """
    frame_list = []
    with open(path) as f:
        lines = f.readlines()

    for l in lines:
        ll = l.split(",")
        frame = f"f_{int(ll[0]) - 1}"
        frame_list.append(
            [
                frame,
                float(ll[2]),
                float(ll[3]),
                float(ll[2]) + float(ll[4]),
                float(ll[3]) + float(ll[5]),
                ll[6],
            ]
        )

    return frame_list


def bounding_box_visualization(path, gt_boxes, predicted_boxes, video_capture, frame_id, iou_scores):
    n_frame = int(frame_id.split('_')[-1])
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, n_frame - 1)
    res, frame = video_capture.read()
    # Draw the ground truth boxes
    for box in gt_boxes[frame_id]:
        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
    # Draw the predicted boxes
    for box in predicted_boxes[frame_id]:
        cv2.rectangle(frame, (int(box[1]), int(box[2])), (int(box[3]), int(box[4])), (0, 0, 255), 2)
    # put text mIOU of frame
    cv2.putText(
        frame,
        f"IoU score: {iou_scores[n_frame]}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    # put text number of frame
    cv2.putText(frame, f"Frame: {n_frame}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imwrite(f'{path}/{frame_id}.png', frame)

    ret, frame = video_capture.read()


def noise_reduction(frame):
    #frame = cv2.erode(frame, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    #frame = cv2.dilate(frame,  cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))

    kernel = np.ones((2, 2), np.uint8)
    seg = cv2.erode(frame, kernel, iterations=1)
    kernel = np.ones((2,4), np.uint8)
    seg = cv2.dilate(seg, kernel, iterations=1)

    seg = cv2.morphologyEx(seg, cv2.MORPH_OPEN, np.ones((10, 5), np.uint8))
    seg = cv2.morphologyEx(seg, cv2.MORPH_CLOSE, np.ones((5, 7), np.uint8))


    seg = cv2.morphologyEx(seg, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    seg = cv2.morphologyEx(seg, cv2.MORPH_CLOSE, np.ones((3, 7), np.uint8))


    return seg

def findBBOX(mask):

    minH = 50
    maxH =  1080/2 # 1080--> height frame
    minW = 100#120
    maxW =1920/2# 1920--> width frame

    contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    box = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if minW < w < maxW and minH < h < maxH:
            if 0.2 < w/h < 10:
                box.append([x, y, x + w, y + h])

    return box


if __name__ == "__main__":
    # Set the parent directory of your current directory
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), "../.."))

    # Set the relative path to the XML file
    relative_path = "dataset/ai_challenge_s03_c010-full_annotation.xml"

    # Get the absolute path of the XML file
    path = os.path.join(parent_dir, relative_path)

    # Print the absolute path
    print(path)
    frame_dict = load_from_xml(path)
    print(frame_dict)
