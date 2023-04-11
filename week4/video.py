import pickle

import cv2
import numpy as np
from skimage import io


def video(det_boxes, method):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_out = cv2.VideoWriter("C:/Users/AnaHarris/Documents/MASTER/M6/project/lab4/" + f"{method}.mp4", fourcc, 10,
                                (1920, 1080))
    tracker_colors = {}

    for frame_id in det_boxes:
        fn = f'C:/Users/AnaHarris/Documents/MASTER/M6/project/dataset/frames/{frame_id}.jpg'
        im = io.imread(fn)
        frame_boxes = det_boxes[frame_id]

        for box in frame_boxes:
            track_id = box[-1]
            if track_id not in tracker_colors:
                tracker_colors[track_id] = np.random.rand(3)
            color = tracker_colors[track_id]
            color = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))

            cv2.rectangle(im, (int(box[1]), int(box[2])), (int(box[3]), int(box[4])), color, 2)
            cv2.putText(im, str(track_id), (int(box[1]), int(box[2])), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2,
                        cv2.LINE_AA)

        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        video_out.write(im)
    video_out.release()


method = 'maskflownet'
detections = pickle.load(open(f'C:/Users/AnaHarris/Documents/MASTER/M6/project/lab4/tracking_{method}.pkl', 'rb'))
video(detections, method)
