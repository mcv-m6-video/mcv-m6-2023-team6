dataset_path = "/export/home/group03/dataset/aic19-track1-mtmc-train/train"
tracking_path = "/export/home/group03/mcv-m6-2023-team6/week5/Results/trackings/MTSC"
results_path = "/export/home/group03/mcv-m6-2023-team6/week5/Results/trackings/MTMC"

import argparse
import os
import cv2
import pickle as pkl
import itertools
from collections import defaultdict
import numpy as np
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize
import torchvision
from torch import nn
import torch
import random

class EmbeddingNetImage(nn.Module):
    def _init_(self, weights, dim_out_fc, network_image = 'RESNET'):   # dim_out_fc = 'as_image' or 'as_text'
        super(EmbeddingNetImage, self)._init_()
        
        self.network_image = network_image
        
        if network_image == 'RESNET50':
            self.model = torchvision.models.resnet50(pretrained=True)
            in_features = self.model.fc.in_features
            self.model.fc = nn.Identity()
            
        elif network_image == 'RESNET101':
            self.model = torchvision.models.resnet101(pretrained=True)
            in_features = self.model.fc.in_features
            self.model.fc = nn.Identity()
        
        # self.fc = nn.Linear(in_features, dim_out_fc)
    def forward(self, x):
        output = self.model(x)   # 2048
        return output
   
seq = "S01"

def crop_from_detection(box,frame):
    image = cv2.cvtColor(cv2.imread(frame),cv2.COLOR_BGR2HSV)
    cropped = image[box[1]:box[3], box[0]:box[2]]
    return cropped

def compute_mr_histogram(img, splits=(1, 1), bins=256,sqrt=False):
    num_splits_x, num_splits_y = splits
    small_img_height = img.shape[0] // num_splits_x
    small_img_width = img.shape[1] // num_splits_y

    histograms = []

    hist = np.array([
                    np.histogram(
                        img[..., channel],
                        bins=bins,
                        density=True
                    )[0] for channel in range(img.shape[2])
                ])
    histograms.append(hist.ravel())
    
    for i in range(num_splits_x):
        for j in range(num_splits_y):
            small_img = img[i * small_img_height : (i + 1) * small_img_height,
                            j * small_img_width : (j + 1) * small_img_width]

            if len(small_img.shape) == 3:
                small_hist = np.array([
                    np.histogram(
                        small_img[..., channel],
                        bins=bins,
                        density=True
                    )[0] for channel in range(small_img.shape[2])
                ])
                histograms.append(small_hist.ravel())
            else:
                raise ValueError("Image should have more than one channel")

    histograms = [np.sqrt(hist) if sqrt else hist for hist in histograms]
    return np.concatenate(histograms, axis=0)

def create_embeddings():
    tracks_embeddings = defaultdict(list)
    sequence_path = os.path.join(dataset_path,seq)

    camaras = ['c001']
    #for c in os.listdir(sequence_path):
    for c in camaras:
        print(c)

        frames_path = os.path.join(sequence_path,c,'frames')
        cap = cv2.VideoCapture(os.path.join(sequence_path,c,'vdo.avi'))
        fps = cap.get(cv2.CAP_PROP_FPS)

        tracking_boxes = pkl.load(open(f'{tracking_path}/{c}.pkl','rb'))

        tracking_boxes_sorted_id = defaultdict(list)
        
        method = 'resnet'
        if method == 'resnet':
            model = EmbeddingNetImage(weights=None, dim_out_fc=2048, network_image='RESNET101')
            model.eval()
            model = nn.DataParallel(model)
            model = model.module

        # MULTIPROCESSING JOHNNY TODO:
        for frame,track in tqdm(tracking_boxes.items()):
            if frame == 186:
                break
            
            for b in track:
                id = int(b[-1])
                bbox = b[1:-1]
                box = [int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])]

                cropped = crop_from_detection(box,f'{frames_path}/{frame}.jpg')
                resized = cv2.resize(cropped,[150,150])

                if method == 'histogram':
                    features = compute_mr_histogram(resized,(2,2))
                elif method == 'resnet':
                    tensor = torch.from_numpy(resized).unsqueeze(0)
                    tensor = tensor.permute(0,3,1,2)
                    features = model((tensor.float()/255.0))

                tracking_boxes_sorted_id[id].append(features)

                
        for id,features in tracking_boxes_sorted_id.items():
            # Mean of the histograms
            # mean_histogram_tracks[c,id] = np.mean(histograms,axis=0)
            # TODO: pick 5 random histograms
            # print('TODO: pick 5 random histograms')
            tracks_embeddings[c,id] = features[0]
            


    with open(f'{results_path}/embedings_resnet_c001.pkl','wb') as h:
        pkl.dump(tracks_embeddings,h,protocol=pkl.HIGHEST_PROTOCOL)



create_embeddings()
cameras_dict = pkl.load(open(f'{results_path}/embedings_resnet_c001.pkl','rb'))

new_dict = {}
for c_id,data in cameras_dict.items():
    new_data = np.concatenate(data.detach().numpy())
    new_dict[c_id] = new_data 


normalized = normalize(np.stack(new_dict.values()))
clustering = DBSCAN(eps=0.57,min_samples=1).fit(normalized) 


groups = defaultdict(list)
for id, label in zip(new_dict.keys(), clustering.labels_):
    groups[label].append(id)
groups = list(groups.values())  


tracking_boxes = pkl.load(open(f'{tracking_path}/c001.pkl','rb'))
"""
results = defaultdict(list)
for global_id, group in enumerate(groups):
    for cam, id in group:
        track = tracking_boxes[id]
        for det in track:
        results[cam].append(track)"""