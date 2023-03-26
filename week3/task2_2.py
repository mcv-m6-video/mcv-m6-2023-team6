
from __future__ import print_function
import matplotlib.pyplot as plt  
import matplotlib.patches as patches
from IPython import display as dp
import numpy as np
from skimage import io
import os
import time
import cv2
from sort.sort import *
from utils.util import load_from_txt,discard_overlaps


def traking(display):

    fileDetections = "/ghome/group03/dataset/AICity_data/train/S03/c010/det/det_mask_rcnn.txt"

    colours = np.random.rand(32,3) #used only for display
    frame_boxes = load_from_txt(fileDetections,threshold=0.5) #load detections
  
    total_time = 0.0
    total_frames = 0
    out = []

    if display:
        plt.ion() # for iterative display
        fig, ax = plt.subplots(1, 2,figsize=(20,20))

    mot_tracker = Sort() #create instance of the SORT tracker

    for frame_id in frame_boxes: # all frames in the sequence

        dets = frame_boxes[frame_id]
        dets = discard_overlaps(dets)

        total_frames += 1

        if display:
            fn = f'/ghome/group03/dataset/AICity_data/train/S03/c010/frames/{frame_id}.jpg'
            im =io.imread(fn)
            ax.imshow(im)
            ax.axis('off')
            ax.set_title('Tracked Targets')

        start_time = time.time()
        dets = np.array(dets)
        trackers = mot_tracker.update(dets)
        cycle_time = time.time() - start_time
        total_time += cycle_time
        
        out.append(trackers)

        for d in trackers:
            if display:
                d = d.astype(np.uint32)
                ax[1].add_patch(patches.Rectangle((d[0],d[1]),d[2]-d[0],d[3]-d[1],fill=False,lw=3,ec=colours[d[4]%32,:]))
                ax[1].set_adjustable('box-forced')


    print("Total Tracking took: %.3f for %d frames or %.1f FPS"%(total_time,total_frames,total_frames/total_time))


if __name__ == "__main__":
    traking(display=False)