import os
import cv2
import imageio
from utils import util
from .metrics import mean_AP_Pascal_VOC
TOTAL_FRAMES_VIDEO = 2141

def rendering_video(cfg, model, frames_modelling, path_results, ai_gt_path, save=True,
                    display=False):
    counter = model.model_background()
    foreground_gif = []

    if not os.path.exists(path_results):
        os.makedirs(path_results)

    det_rects = {}
    gt_rects = util.load_from_xml(ai_gt_path)
    gt_rects = {k: v for k, v in gt_rects.items() if
                int(k.split('_')[-1]) >= frames_modelling}  # remove "training" frames 

    foreground, I = model.compute_next_foreground()

    #det_rects[f'f_{counter}'] = I

    """ det_rects = {}
    gt_rects = util.load_from_xml(ai_gt_path)
    gt_rects = {k: v for k, v in gt_rects.items() if
                int(k.split('_')[-1]) >= frames_modelling}  # remove "training" frames

    gt_rects_detformat = {f: [{'bbox': r, 'conf': 1} for r in v] for f, v in gt_rects.items()} """
   

    counter = frames_modelling
    for i in range(TOTAL_FRAMES_VIDEO):
        while foreground is not None:
            if cfg['display']:
                pass

            
            counter += 1

            ret = model.compute_next_foreground()
            if ret:
                foreground, I = ret
                foreground_gif.append(foreground)  # ADD IMAGE GIF
                # TODO: SOMETHING WITH DETECTIONS
               
                
            else:
                foreground = None

            if counter % 100 == 0:
                print(f"{counter} frames processed...")

            if  counter >= -1:
                break
    
    print(f"DONE! {counter} frames processed")
    print(f"Saved to '{path_results}'")

    
    """ mAP = mean_AP_Pascal_VOC(gt_rects, det_rects)
    print('mAP:', mAP) """
    
    # Save GIF
    if cfg['save']:
        imageio.mimsave(f'{path_results}/foreground.gif', foreground_gif[:200])
