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
matplotlib.use('TkAgg')
from tqdm import tqdm


def traking(display):
    images = []
    current_path = os.path.dirname(os.path.abspath(__file__))
    fileDetections = os.path.join(current_path, "../../dataset/AICity_data/train/S03/c010/det/det_mask_rcnn.txt")

    colours = np.random.rand(32, 3)  # used only for display
    frame_boxes = load_from_txt(fileDetections, threshold=0.5)  # load detections

    total_time = 0.0
    total_frames = 0
    out = []

    if display:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_out = cv2.VideoWriter("./output/" + "video.mp4", fourcc, 10, (1920, 1080))

    mot_tracker = Sort()  # create instance of the SORT tracker

    for frame_id in tqdm(frame_boxes):  # all frames in the sequence

        dets = frame_boxes[frame_id]
        dets = discard_overlaps(dets)

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
            cv2.rectangle(im, (d[0], d[1]), (d[2], d[3]), (0, 255, 0), 2)

        if display:
            cv2.imshow('frame', im)
        video_out.write(im)


    video_out.release()
    print("Total Tracking took: %.3f for %d frames or %.1f FPS" % (total_time, total_frames, total_frames / total_time))


if __name__ == "__main__":
    traking(display=False)
