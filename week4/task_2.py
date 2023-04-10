import argparse
from .datasets import AICityDataset
from task_1_3 import max_iou_tracking
import os
import pickle

def task2(args):
    #dataset = AICityDataset(args.dataset_path, args.sequences)
    for c in os.listdir(args.dataset_path+'/'+args.sequences):
        detections = args.dataset_path+'/'+args.sequences+'/'+c+'/det/det_yolo3.txt'
        frames_path = args.dataset_path+'/'+args.sequences+'/'+c+'/frames/'

        tracking_boxes = max_iou_tracking(detections,'maskflownet',frames_path)

        with open(f'{args.results_path}/tracking_maskflownet_{c}.pkl','wb') as h:
            pickle.dump(tracking_boxes,h,protocol=pickle.HIGHEST_PROTOCOL)





if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_path', type=str, default="/ghome/group03/dataset/aic19-track1-mtmc-train",
                        help='dataset')
    
    parser.add_argument('--sequences', type=str, default="S03", help='sequences')

    parser.add_argument('--results_path', type=str, default='Results/Task2/',
                        help='path to save results in a csv file')
    parser.add_argument('--visualize', type=bool, default=True)

    args = parser.parse_args()


    task2(args)