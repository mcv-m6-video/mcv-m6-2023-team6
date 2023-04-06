import sys
from tqdm import tqdm

sys.path.insert(0,'week3')

import os
import numpy as np
from PIL import Image
from task_1_2 import flow_pyflow,flow_LK
from utils.maskflow import maskflownet
from utils.utils_trackeval import save_txt
import pandas as pd
import argparse
import time
import copy
from week3.utils.util import load_from_txt,discard_overlaps,filter_boxes,iou
from week3.task2_1 import track_memory
from utils.optical_flow import compute_errors,flow_read, HSVOpticalFlow2, opticalFlow_arrows

def max_iou_tracking(path,method,conf_threshold=0.5,iou_threshold=0.5):
    total_time = 0.0
    total_frames = 0

    det_boxes = load_from_txt(path,threshold=conf_threshold)

    track_id = 0
    tracked_objects = {}
    memory = 5
    corrected_csv = {}
    for frame_id in tqdm(det_boxes):
        
        total_frames += 1
        start_time = time.time()
        # REMOVE OVERLAPPING BOUNDING BOXES 
        boxes = det_boxes[frame_id]
        boxes = discard_overlaps(boxes)
        frame_boxes = filter_boxes(boxes)


        # FIRST FRAME, WE INITIALIZE THE OBJECTS ID
        if not tracked_objects:
            for j in range(len(frame_boxes)):
                # We add the tracking object ID at the end of the list  [[frame,x1, y1, x2, y2, conf, track_id]]
                frame_boxes[j].append(track_id)
                tracked_objects[f'{track_id}'] = {'bbox':[frame_boxes[j][1],frame_boxes[j][2],frame_boxes[j][3],frame_boxes[j][4]],'frame':frame_id,'memory':0, 'iou':0}
                track_id += 1           
                
        else:
        
            # FRAME N+1 WE COMPARE TO OBJECTS IN FRAME N
            for i in range(len(frame_boxes)):
                frame_boxes[i][0] = frame_id
                best_iou = 0
                track_id_best = 0
                boxA = [frame_boxes[i][1],frame_boxes[i][2],frame_boxes[i][3],frame_boxes[i][4]]

                for data in previous_tracked_objects.items():
                    id,boxB = data
                    iou_score,_ = iou(boxA,boxB['bbox'])

                    if iou_score > best_iou and iou_score >= iou_threshold:
                        best_iou = iou_score
                        track_id_best = id

                if track_id_best == 0 and best_iou == 0:
                    frame_boxes[i].append(track_id)
                    tracked_objects[f'{track_id}']  = {'bbox':boxA,'frame':frame_id,'memory':0,'iou':best_iou}
                    track_id += 1
                   
                    
                else:
                    if tracked_objects[f'{track_id_best}']['frame'] == frame_id:
                        # CHECK IF THERE IS AN OBJECT WITH THE SAME ID IN THAT FRAME AND CHOOSE THE ONE WITH HIGHEST IOU
                        if best_iou > tracked_objects[f'{track_id_best}']['iou']:
                            tracked_objects[f'{track_id}'] = {'bbox':tracked_objects[f'{track_id_best}']['bbox'],'frame':frame_id,'memory':0,'iou':best_iou}
                            wrong_id =  [i for i,det in enumerate(frame_boxes) if det[-1] == track_id_best][0]
                            frame_boxes[wrong_id][-1] = track_id
                            track_id += 1

                            frame_boxes[i].append(track_id_best)
                            tracked_objects[f'{track_id_best}']['bbox']= boxA
                            previous_f = tracked_objects[f'{track_id_best}']['frame']

                            # CHECK IF OBJECTS APPEAR CONSECUTIVE
                            if frame_id - previous_f == 1:
                                tracked_objects[f'{track_id_best}']['memory'] = tracked_objects[f'{track_id_best}']['memory'] + 1
                            tracked_objects[f'{track_id_best}']['frame'] = frame_id
                            tracked_objects[f'{track_id_best}']['iou'] = best_iou

                        else:
                            frame_boxes[i].append(track_id)
                            tracked_objects[f'{track_id}']  = {'bbox':boxA,'frame':frame_id,'memory':0,'iou':best_iou}
                            track_id += 1
               

                    else:
                        frame_boxes[i].append(track_id_best)
                        tracked_objects[f'{track_id_best}']['bbox']= boxA
                        previous_f = tracked_objects[f'{track_id_best}']['frame']

                        # CHECK IF OBJECTS APPEAR CONSECUTIVE
                        if frame_id - previous_f == 1:
                            tracked_objects[f'{track_id_best}']['memory'] = tracked_objects[f'{track_id_best}']['memory'] + 1
                        tracked_objects[f'{track_id_best}']['frame'] = frame_id
                        tracked_objects[f'{track_id_best}']['iou'] = best_iou


        if frame_id == memory:
            track_memory(tracked_objects)
            memory = memory + frame_id 

        previous_tracked_objects = copy.deepcopy(tracked_objects)
        cycle_time = time.time() - start_time
        total_time += cycle_time

    print("Total Tracking took: %.3f for %d frames or %.1f FPS" % (total_time, total_frames, total_frames / total_time))
    return det_boxes

estimate_flow = {
    'pyflow': flow_pyflow,
    'LK': flow_LK,
    'maskflownet': maskflownet,
}

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--gt', type=str, default= "/ghome/group03/dataset/ai_challenge_s03_c010-full_annotation.xml",
                        help='ground truth xml file for object tracking')

    parser.add_argument('--detections', type=str, default="/ghome/group03//export/home/group03/mcv-m6-2023-team6/week3/Results/Task1_5/faster_RCNN/A/bbox_faster_RCNN_A.txt",
                        help='.txt file with the object detection')

    parser.add_argument('--frames_path', type=str, default="/ghome/group03/dataset/AICity_data/train/S03/c010/frames/",
                        help='path to folder containing the images to estimate the object tracking with optical flow')

    parser.add_argument('--results_path', type=str, default='Results/Task1_3/',
                        help='path to save results')
    parser.add_argument('--visualize', type=bool, default=True)

    args = parser.parse_args()
    
    # Get the directory of the current file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Output path for the results
    output_path = os.path.join(current_dir, args.results_path)
    
    # Create the output directory if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    methods = ['pyflow','LK', 'maskflownet']
    
    results = []
    

    # perform grid using the multiple combinations of the parameters using product show progress in tqdm
    for method in methods:
        print('.............Object tracking with optical flow for method: {}....................'.format(method))
        output_path_method = os.path.join(output_path, method)
        
        start = time.time()
        tracking_boxes = max_iou_tracking(args.detections,method)
        #flow = estimate_flow[method](PREV FRAME, ACTUAL FRAME, colType=1) ESTO VA DENTRO DE LA FUNCION DE TRACKING
        save_txt(tracking_boxes,args.results_path)
        end = time.time()
        
