from utils.util import load_from_txt,discard_overlaps,iou,write_to_csv_file,filter_boxes
import cv2
import numpy as np
from skimage import io
from tqdm import tqdm
import time
import copy
import os
import argparse


current_path = os.path.dirname(os.path.abspath(__file__))

def track_memory(tracked_objects):
    delete = []
    for idx in tracked_objects:
        if tracked_objects[idx]['memory'] != tracked_objects[idx]['frame']:
            if tracked_objects[idx]['memory'] <= 5:
                delete.append(idx)
 

    for idx in delete:
        del tracked_objects[idx]

def video(det_boxes,method):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_out = cv2.VideoWriter("./Results/Task_2_1/" + f"{method}.mp4", fourcc, 10, (1920, 1080))
    tracker_colors = {}

    for frame_id in det_boxes:
        fn = current_path + f'/../../dataset/AICity_data/train/S03/c010/frames/{frame_id}.jpg'
        im = io.imread(fn)
        frame_boxes = det_boxes[frame_id]
        

        for box in frame_boxes:
            track_id = box[-1]
            if track_id not in tracker_colors:
                tracker_colors[track_id] = np.random.rand(3)
            color = tracker_colors[track_id]
            color = (int(color[0]*255),int(color[1]*255), int(color[2]*255))

            cv2.rectangle(im, (int(box[1]), int(box[2])), (int(box[3]), int(box[4])),color,2)
            cv2.putText(im, str(track_id), (int(box[1]), int(box[2])), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        video_out.write(im)
    video_out.release()

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
    
    #video(det_boxes,method)
    
    for frame_number, object_list in det_boxes.items():
      f_id = frame_number + 1
      for obj in object_list:
        obj[0] = f_id
        
      corrected_csv[f_id] = object_list
      
    
    return corrected_csv

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--method", required=True, type=str, help="faster_RCNN or retinaNet")

    
    args = parser.parse_args()
    
    grid_conf = [0.60,0.65,0.70,0.75,0.80,0.85,0.90]

    method = args.method
    
    for c in grid_conf:
      tracking_boxes = max_iou_tracking(os.path.join(current_path, f"./Results/Task1_5/{method}/A/bbox_{method}_A.txt"),method,conf_threshold=float(c))
      write_to_csv_file(f"./Results/Task_2_1/task_2_1_{method}_thr{str(c)[-2:]}.csv",tracking_boxes)
    
  
    
    
    
    