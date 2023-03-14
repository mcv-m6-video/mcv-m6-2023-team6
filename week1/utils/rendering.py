import itertools

import cv2
import numpy as np

from week1.utils.metrics import mean_AP_Pascal_VOC
from week1.utils.utils import load_from_xml, load_from_txt


# Rendering Video AICity Challenge 2023

def group_by_frame(predicted_boxes):
    """Group the detected boxes by frame_id as a dictionary"""
    predicted_boxes.sort(key=lambda x: x[0])
    predicted_boxes = itertools.groupby(predicted_boxes, key=lambda x: x[0])
    predicted_boxes = {k: list(v) for k, v in predicted_boxes}
    return predicted_boxes


def rendering_video(path, annotations, predicted_boxes, video_capture, save=True, display=False):
    """Create a video with the IoU score for each frame"""
    # Group the detected boxes by frame_id as a dictionary
    gt_boxes, total = load_from_xml(annotations)
    predicted_boxes = load_from_txt(predicted_boxes)
    predicted_boxes_group = group_by_frame(predicted_boxes)
    # Get the IoU score for each frame in format dict {frame_id: [iou_score1, iou_score2, ...]}
    mIOU, mIOU_frame, AP = mean_AP_Pascal_VOC(gt_boxes, total, predicted_boxes, iou_th=0.5)
    # Get the frame_id list
    frames_id = list(mIOU_frame.keys())
    # Sort the frames list
    frames_id.sort(key=lambda x: int(x.split('_')[1]))
    frames_num = [int(frame.split('_')[1]) for frame in frames_id]
    # Get the IoU score list
    iou_scores = [np.mean(mIOU_frame[frame]) for frame in frames_id]

    # Open the video
    video_capture = cv2.VideoCapture(video_capture)
    # Get the video fps
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    # Get the video width
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    # Get the video height
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))

    # Loop through each frame
    for i, frame_id in enumerate(frames_id):
        # Read the frame
        ret, frame = video_capture.read()
        if ret:
            # Convert the frame to RGB
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Draw the ground truth boxes
            for box in gt_boxes[frame_id]:
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
            # Draw the predicted boxes
            for box in predicted_boxes_group[frame_id]:
                cv2.rectangle(frame, (int(box[1]), int(box[2])), (int(box[3]), int(box[4])), (0, 0, 255), 2)
            # Draw the IoU score
            iou_score = round(iou_scores[i], 2)
            cv2.putText(frame, f"IoU score: {iou_score}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                        cv2.LINE_AA)
            # put text number of frame
            cv2.putText(frame, f"Frame: {frames_num[i]}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                        cv2.LINE_AA)
            # put fps
            cv2.putText(frame, f"FPS: {fps}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            # Write the frame to the video
            out.write(frame)
            if display:
                cv2.imshow('frame', frame)
                k = cv2.waitKey(wait_time)
                if k == ord('q'):
                    break
                elif k == ord('s'):
                    cv2.imwrite(f'save_{frames_num[i]}.png', frame)
                elif k == ord('p'):
                    wait_time = int(not (bool(wait_time)))
    if save:
        # Release the VideoWriter object
        out.release()
