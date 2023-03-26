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
        plt.ion()  # for iterative display
        fig, ax = plt.subplots(1, 2, figsize=(20, 20))
        fig.show()

    mot_tracker = Sort()  # create instance of the SORT tracker

    for frame_id in tqdm(frame_boxes):  # all frames in the sequence

        dets = frame_boxes[frame_id]
        dets = discard_overlaps(dets)

        total_frames += 1

        if display:
            fn = current_path + f'/../../dataset/AICity_data/train/S03/c010/frames/{frame_id}.jpg'
            im = io.imread(fn)
            ax[0].axis('off')
            ax[0].set_title('Tracked Targets')
            ax[0].imshow(im)
            # fig.canvas.draw()



        start_time = time.time()
        dets = np.array(dets)
        trackers = mot_tracker.update(dets)
        cycle_time = time.time() - start_time
        total_time += cycle_time

        out.append(trackers)

        for d in trackers:
            if display:
                d = d.astype(np.uint32)
                ax[1].add_patch(patches.Rectangle((d[0], d[1]), d[2] - d[0], d[3] - d[1], fill=False, lw=3,
                                                  ec=colours[d[4] % 32, :]))
                ax[1].set_adjustable('box')

        # fig.canvas.draw()
        #canvas = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        # canvas = canvas.reshape(fig.canvas.get_width_height(physical=True)[::-1] + (3,))
        # images.append(canvas)
        plt.savefig(f'./output/{frame_id}.jpg')


    # imageio.mimsave("./output" + 'iou.gif', images)

        # cv2.imwrite(f'./output/{frame_id}.jpg', fig.canvas.buffer_rgba())



    print("Total Tracking took: %.3f for %d frames or %.1f FPS" % (total_time, total_frames, total_frames / total_time))


if __name__ == "__main__":
    traking(display=True)
