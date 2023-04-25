import argparse
import os
import cv2
import pickle as pkl
import pprint

from utils import max_iou_tracking, max_iou_tracking_OF, util

use_OF = False
dataset_path = "/export/home/group03/dataset/aic19-track1-mtmc-train/train/"
detections_path = "/export/home/group03/mcv-m6-2023-team6/week5/Results/detections/"

if use_OF:
    results_path = "/export/home/group03/mcv-m6-2023-team6/week5/Results/trackings_OF/"
else:
    results_path = "/export/home/group03/mcv-m6-2023-team6/week5/Results/trackings/"

sequences = ["S01","S04"]

for s in sequences:
    sequence_path = os.path.join(dataset_path,s)
    for c in os.listdir(sequence_path):
        det_path = os.path.join(detections_path,c+'.txt')
        frames_path = os.path.join(sequence_path,c,'frames')
        cap = cv2.VideoCapture(os.path.join(sequence_path,c,'vdo.avi'))
        fps = cap.get(cv2.CAP_PROP_FPS)
        det_boxes = util.load_from_txt(det_path)

        if use_OF:
            tracking_boxes = max_iou_tracking_OF.max_iou_tracking_withoutParked(det_boxes,frames_path,fps)
        else:
            tracking_boxes = max_iou_tracking.max_iou_tracking_withoutParked(det_boxes,frames_path,fps)

        with open(f'{results_path}/{c}.pkl','wb') as h:
            pkl.dump(tracking_boxes,h,protocol=pkl.HIGHEST_PROTOCOL)

        #open pkl file
        
        

util.convert_pkl_to_txt(results_path,results_path)