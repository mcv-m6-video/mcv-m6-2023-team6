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



dataset_path = "/export/home/group03/dataset/aic19-track1-mtmc-train/train"
tracking_path = "/export/home/group03/mcv-m6-2023-team6/week5/Results/trackings/MTSC"
results_path = "/export/home/group03/mcv-m6-2023-team6/week5/Results/trackings/MTMC"

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
    mean_histogram_tracks = defaultdict(list)
    sequence_path = os.path.join(dataset_path,seq)

    for c in os.listdir(sequence_path):
        print(c)

        frames_path = os.path.join(sequence_path,c,'frames')
        cap = cv2.VideoCapture(os.path.join(sequence_path,c,'vdo.avi'))
        fps = cap.get(cv2.CAP_PROP_FPS)

        tracking_boxes = pkl.load(open(f'{tracking_path}/{c}.pkl','rb'))

        tracking_boxes_sorted_id = defaultdict(list)
        
        method = 'histogram'
        if method == 'resnet':
            model = EmbeddingNetImage(weights=None, dim_out_fc=2048, network_image='RESNET101')
            model.eval()
            model.cuda()
            model = nn.DataParallel(model)
            model = model.module

        # MULTIPROCESSING JOHNNY TODO:
        for frame,track in tqdm(tracking_boxes.items()):
            for b in track:
                i = 1
                if len(track) >= 5:

                    id = int(b[-1])
                    bbox = b[1:-1]
                    box = [int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])]

                    cropped = crop_from_detection(box,f'{frames_path}/{frame}.jpg')
                    resized = cv2.resize(cropped,[150,150])

                    if method == 'histogram':
                        histogram = compute_mr_histogram(resized,(3,3))
                    elif method == 'resnet':
                        histogram 

                    tracking_boxes_sorted_id[id].append(histogram)
                    i +=1
                    if i == 6:
                        break

                
        for id,histograms in tracking_boxes_sorted_id.items():
            mean_histogram_tracks[c,id] = histograms



    with open(f'{results_path}/cameras_embeddings_S01.pkl','wb') as h:
        pkl.dump(mean_histogram_tracks,h,protocol=pkl.HIGHEST_PROTOCOL)



#create_embeddings()

cameras_dict = pkl.load(open(f'{results_path}/cameras_embeddings_S01.pkl','rb'))

# normalized = normalize(np.stack(cameras_dict.values()))
# clustering = DBSCAN(eps=3,min_samples=2).fit(normalized)
clustering = DBSCAN(eps=3,min_samples=2).fit(np.stack(cameras_dict.values()))


groups = defaultdict(list)
for id, label in zip(cameras_dict.keys(), clustering.labels_):
    groups[label].append(id)
groups = list(groups.values())


""" results = defaultdict(list)
for global_id, group in enumerate(groups):
    for cam, id in group:
        track = tracks_by_cam[cam][id]
        for det in track:
            det.id = global_id
        results[cam].append(track) """
                

        
        

    











        
        
