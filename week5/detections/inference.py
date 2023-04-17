from ultralytics import YOLO
import argparse
import os
from utils.util import *

TRAIN = ['S01','S04']
TEST = ['S03']

def main(data_path,results_path):
    for seq in TRAIN:
        for c in os.listdir(data_path+'/'+seq):
            cd =data_path+'/'+seq+'/'+c
            gt = load_from_txt(cd+'/gt/gt.txt')
            frames = cd+'/frames'
            model = YOLO()
           

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Detections: Fine-tuning')
    parser.add_argument('-d', type=str, default='/export/home/group03/dataset/aic19-track1-mtmc-train/train/', help='Dataset directory')
    parser.add_argument('-r', type=str, default='/Results/detections/', help='Path to save results')

    args = parser.parse_args()
    
    # Get the directory of the current file
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Output path for the results
    output_path = os.path.join(current_dir, args.r)

    # Create the output directory if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    main(args.d,output_path)