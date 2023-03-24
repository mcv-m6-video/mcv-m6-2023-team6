import os
import random

import cv2
import numpy as np
import pycocotools.mask as mask_utils
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures.boxes import BoxMode
from detectron2.utils.visualizer import Visualizer
from tqdm import tqdm
import xmltodict


def parse_xml_bb(path_xml, includeParked = True):
    """
    Input:
        - Path to xml in Pascal VOC format annotations
    Output format:
        dict[frame_id] = [{'bbox':[x1, y1, x2, y2], 'conf': 1, 'id': -1}]
    """

    with open(path_xml, 'rb') as f:
        xml_dict = xmltodict.parse(f)

    frame_dict = {}

    for track in xml_dict['annotations']['track']:

        if track['@label'] != 'car':
            continue

        track_id = int(track['@id'])

        for bbox in track['box']:

            frame = f"f_{bbox['@frame']}"

            if bbox['attribute']['#text'] == 'true' and includeParked == False:
                if frame not in frame_dict:
                    frame_dict[frame] = []
                continue

            if frame not in frame_dict:
                frame_dict[frame] = []
                
            frame_dict[frame].append({
                'conf': 1,
                'bbox': [float(bbox['@xtl']), float(bbox['@ytl']), float(bbox['@xbr']), float(bbox['@ybr'])],
                'id': track_id
            })

    return frame_dict



def get_CityAI_dicts(subset):
    images = "/ghome/group03/dataset/AICity_data/train/S03/c010/frames"
    annotations = "/ghome/group03/dataset/ai_challenge_s03_c010-full_annotation.xml"
    gt_bb = parse_xml_bb(annotations)

    total_frames = len(os.listdir(images))

    if subset == "train":
        # create a list of the first 25% of the frames
        start = 0
        end = int(total_frames*0.25)
        
    elif subset == "val":
        # create a list of the last 75% of the frames
        start = int(total_frames*0.25) + 1
        end = total_frames


    dataset_dicts = []
 
    for seq_id in range(start,end):
            
        record = {}

        filename = os.path.join(images, str(seq_id) + ".jpg"	)

        record["file_name"] = filename
        record["image_id"] = seq_id
        record["height"] = 1080
        record["width"] = 1920

        objs = []
        gt = gt_bb[f'f_{seq_id}']

        for obj in gt:
            bb = obj['bbox']
             
            obj = {
                "bbox": list(map(int, bb)),
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": 0,
            }
            objs.append(obj)

        record["annotations"] = objs

        dataset_dicts.append(record)

    return dataset_dicts
