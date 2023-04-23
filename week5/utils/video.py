import os
import pickle

import cv2
import numpy as np
from skimage import io
from tqdm import tqdm


def video(det_boxes, fps, seq, camera,output_path):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_out = cv2.VideoWriter(f"{output_path}/{seq}_{camera}.mp4", fourcc, fps, (1920, 1080))
    tracker_colors = {}

    #get the frames in order

    frames = os.listdir(f'/ghome/group03/dataset/aic19-track1-mtmc-train/train/{seq}/{camera}/frames')
    frames = sorted(frames, key=lambda x: int(x.split('.')[0]))
    
    # current path file
    current_path = os.path.dirname(os.path.abspath(__file__))
    for frame_id in tqdm(frames):
        fn = f'/ghome/group03/dataset/aic19-track1-mtmc-train/train/{seq}/{camera}/frames/{frame_id}'
        im = io.imread(fn)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        id = int(frame_id.split('.')[0])
        if id in det_boxes.keys():
            frame_boxes = det_boxes[id]

            for box in frame_boxes:
                track_id = box[-1]
                if track_id not in tracker_colors:
                    tracker_colors[track_id] = np.random.rand(3)
                color = tracker_colors[track_id]
                color = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))

                cv2.rectangle(im, (int(box[1]), int(box[2])), (int(box[3]), int(box[4])), color, 2)
                cv2.putText(im, str(track_id), (int(box[1]), int(box[2])), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2,
                            cv2.LINE_AA)

        video_out.write(im)
    video_out.release()

"""

cameras = ['c010','c011','c012','c013', 'c014', 'c015']
output_path = '/ghome/group03/mcv-m6-2023-team6/week5/Results/videos'
for camera in cameras:
    fps = 10
    detections = pickle.load(open(f'/ghome/group03/mcv-m6-2023-team6/week5/Results/trackings/{camera}.pkl', 'rb'))
    seq = 'S03'
    
    video(detections, fps,seq,camera, output_path)
"""

