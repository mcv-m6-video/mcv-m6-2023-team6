import os
import cv2
import imageio
import numpy as np
from utils import util
from .metrics import mean_AP_Pascal_VOC,compute_confidences_ap

TOTAL_FRAMES_VIDEO = 2141

def rendering_video(cfg, model, frames_modelling, path_results, ai_gt_path, save=True,
                    display=False):
    model.model_background()
    foreground_gif = []
    foreground_gif_boxes = []

    cap = cv2.VideoCapture(cfg["paths"]["video_path"])

    if not os.path.exists(path_results):
        os.makedirs(path_results)

    det_rects = []
    gt_rects = util.load_from_xml(ai_gt_path)
    gt_rects = {k: v for k, v in gt_rects.items() if
                int(k.split('_')[-1]) >= frames_modelling}  

    foreground, I = model.compute_next_foreground()
    foreground = util.noise_reduction(foreground)
    frame_bbox = util.findBBOX(foreground)
    #det_rects[f'f_{counter}'] = frame_bbox
    
    det_rects.append([f'f{counter}',frame_bbox])
    counter = frames_modelling
    for i,frames_id in enumerate(gt_rects):
        while foreground is not None:
            if cfg['display']:
                pass

            counter += 1

            ret = model.compute_next_foreground()
            if ret:
                foreground, I = ret
                foreground = util.noise_reduction(foreground)
                foreground_gif.append(foreground)  # ADD IMAGE GIF
                frame_bbox = util.findBBOX(foreground)
                det_rects.append([f'f{frames_id}',frame_bbox])
                #det_rects[f'f_{frames_id}'] = frame_bbox
                # GT bounding box
                for box in gt_rects[frames_id]:
                    cv2.rectangle(foreground, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
                # Detected bounding box
                for box in frame_bbox:
                    cv2.rectangle(foreground, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
                foreground_gif_boxes.append(foreground)  # ADD IMAGE GIF
               
                
            else:
                foreground = None

            if counter % 100 == 0:
                print(f"{counter} frames processed...")

            if  counter >= -1:
                break
    
    print(f"DONE! {counter} frames processed")
    print(f"Saved to '{path_results}'")
    
    mAP = compute_confidences_ap(gt_rects, len(gt_rects),det_rects)
    print('mAP:', mAP)
    
    # Save GIF
    if cfg['save']:
        imageio.mimsave(f'{path_results}/denoised3_foreground.gif', foreground_gif[:200])
        imageio.mimsave(f'{path_results}/denoised3_foreground.gif', foreground_gif_boxes[:200])
