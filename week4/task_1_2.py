import os
import sys
import time 
import cv2 
import argparse
from PIL import Image
import numpy as np

import pandas as pd

from utils.optical_flow import compute_errors,flow_read, HSVOpticalFlow2, opticalFlow_arrows

# Clone Repo
# 
import pyflow.pyflow as pyflow


# Clone Repo
# https://github.com/microsoft/MaskFlownet
# Set path to MaskFlownet in utils/maskflow.py
from utils.maskflow import maskflownet



# sys.path.append("RAFT")
# from demo import flow_raft



def flow_pyflow(img_prev, img_next, colType=0):
    img_prev = img_prev.astype(float) / 255.
    img_next = img_next.astype(float) / 255.

    # Flow Options:
    alpha = 0.012
    ratio = 0.75
    minWidth = 20
    nOuterFPIterations = 7
    nInnerFPIterations = 1
    nSORIterations = 30
    colType = colType  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))
    
    if colType == 0:
        img_prev = np.expand_dims(img_prev, axis=2)
        img_next = np.expand_dims(img_next, axis=2)

    u, v, _ = pyflow.coarse2fine_flow(
        img_prev, img_next, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
        nSORIterations, colType)
    flow = np.concatenate((u[..., None], v[..., None]), axis=2)

    return flow


def flow_LK(img_prev, img_next, colType=0):

    if colType == 1:
        img_prev = cv2.cvtColor(img_prev, cv2.COLOR_BGR2GRAY)
        img_next = cv2.cvtColor(img_next, cv2.COLOR_BGR2GRAY)

    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Take all pixels
    height, width = img_prev.shape[:2]
    p0 = np.array([[x, y] for y in range(height) for x in range(width)], dtype=np.float32).reshape((-1, 1, 2))

    p1, st, err = cv2.calcOpticalFlowPyrLK(img_prev, img_next, p0, None, **lk_params)
    p0 = p0.reshape((height, width, 2))
    p1 = p1.reshape((height, width, 2))
    st = st.reshape((height, width))

    flow = p1 - p0
    flow[st == 0] = 0

    return flow


estimate_flow = {
    'pyflow': flow_pyflow,
    'LK': flow_LK,
    'maskflownet': maskflownet,
    # 'raft': raft
}


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--gt_path', type=str, default= "/ghome/group03/dataset/OpticalFlow/data_stereo_flow/",
                        help='path to ground truth file for optical flow')

    parser.add_argument('--frames_path', type=str, default="/ghome/group03/dataset/OpticalFlow/frames/colored_0/",
                        help='path to folder containing the images to estimate the optical flow')

    parser.add_argument('--results_path', type=str, default='Results/Task1_2/',
                        help='path to save results in a csv file')
    parser.add_argument('--visualize', type=bool, default=True)

    args = parser.parse_args()
    
    # Get the directory of the current file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Output path for the results
    output_path = os.path.join(current_dir, args.results_path)
    
    # Create the output directory if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)


    img_10 = np.array(Image.open(os.path.join(args.frames_path, '000045_10.png')))
    img_11 = np.array(Image.open(os.path.join(args.frames_path, '000045_11.png')))

    # methods = ['pyflow', 'LK']
    methods = ['pyflow','LK', 'maskflownet']
    
    results = []
    
    flow_gt = flow_read(os.path.join(args.gt_path, '000045_10.png'))

    # perform grid using the multiple combinations of the parameters using product show progress in tqdm
    for method in methods:
        print('.................Estimating flow for method: {}....................'.format(method))
        output_path_method = os.path.join(output_path, method)
        
        start = time.time()
        flow = estimate_flow[method](img_10, img_11, colType=1)
        end = time.time()
        
        msen, pepn = compute_errors(flow, flow_gt, threshold=3, save_path=output_path_method+'/')

        #visualize_flow
        if args.visualize:
            opticalFlow_arrows(img_10, flow_gt, flow, save_path=output_path_method+'/')
            HSVOpticalFlow2(flow, save_path=output_path_method+'/')

        results.append([method, msen, pepn, end-start])

    df = pd.DataFrame(results, columns=['method' , 'msen', 'pepn', 'runtime'])

    print(df)
    

    df.to_csv(output_path + 'results.csv', index=False)