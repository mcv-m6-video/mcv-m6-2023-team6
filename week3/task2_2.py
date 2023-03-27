from __future__ import print_function

import imageio
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from IPython import display as dp
import numpy as np
from skimage import io
import os
import time
import cv2
from sort.sort import *
from utils.util import load_from_txt, discard_overlaps
import matplotlib
from tqdm import tqdm


def traking(display):
    images = []
    current_path = os.path.dirname(os.path.abspath(__file__))
    fileDetections = os.path.join(current_path, "../../dataset/AICity_data/train/S03/c010/det/det_mask_rcnn.txt")

    colours = np.random.rand(100, 3)  # used only for display
    frame_boxes = load_from_txt(fileDetections, threshold=0.5)  # load detections

    total_time = 0.0
    total_frames = 0
    out = []

    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_out = cv2.VideoWriter("./output_task2_2/" + "task2_2.mp4", fourcc, 10, (1920, 1080))

    mot_tracker = Sort()  # create instance of the SORT tracker
    tracker_colors = {}

    for frame_id in tqdm(frame_boxes):  # all frames in the sequence

        dets = frame_boxes[frame_id] # each box is [frame,x1, y1, x2, y2, conf]
        dets = discard_overlaps(dets)
        # from each box we extract only the x1, y1, x2, y2
        dets = [[d[1], d[2], d[3], d[4]] for d in dets]

        total_frames += 1
        fn = current_path + f'/../../dataset/AICity_data/train/S03/c010/frames/{frame_id}.jpg'
        im = io.imread(fn)
        start_time = time.time()
        dets = np.array(dets)

        trackers = mot_tracker.update(dets)
        cycle_time = time.time() - start_time
        total_time += cycle_time

        out.append(trackers)

        for d in trackers:
            d = d.astype(np.uint32)
            tracker_id = d[4]
            if tracker_id not in tracker_colors:
                # generate a new random color for this tracker
                tracker_colors[tracker_id] = np.random.rand(3)
            color = tracker_colors[tracker_id]
            #color array to tuple
            color = (int(color[0]*255), int(color[1]*255), int(color[2]*255))
            
            cv2.rectangle(im, (d[0], d[1]), (d[2], d[3]),color,2)
            cv2.putText(im, str(tracker_id), (d[0], d[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        if display:
            cv2.imshow('frame', im)
        video_out.write(im)


    video_out.release()
    print("Total Tracking took: %.3f for %d frames or %.1f FPS" % (total_time, total_frames, total_frames / total_time))
    print(out)


if __name__ == "__main__":
    traking(display=False)
