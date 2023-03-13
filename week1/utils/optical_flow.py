#Optical flow estimations using the Lucas-Kanade algorithm.
# Sequences 45 and 157 (image_0) from the KITTI dataset.
# Only 1 estimation / sequence (2 frames!)
# Check the KITTI website for code to read results (dense motion vectors)
import cv2
import numpy as np
import matplotlib.pyplot as plt


def flow_read(filename):
    """Read optical flow from png file (adapted version from KITTI development tools kit).
    Args:
        filename: path to optical flow png file.
    Returns:
        F: optical flow.
    """
    I = cv2.imread(filename,cv2.IMREAD_UNCHANGED).astype(np.double)

    F_u = (I[:,:,2]-2**15)/64
    F_v = (I[:,:,1]-2**15)/64

    #check if there exists a valid GT flow for that pixel (1: True, 0: False)
    F_valid = I[:,:,0] 
    F_valid[F_valid>1] = 1

    F_u[F_valid==0] = 0
    F_v[F_valid==0] = 0

    F = np.dstack((F_u, F_v, F_valid))
    return F

def compute_errors(pred_flow, gt_flow, threshold, plots=False):
    """Compute the mean squared error in Non-occluded (MSEN) areas and the percentage of erroneous pixels (PEPN).
       Visualize the results if plots = True.
    Args:
        pred_flow: predicted optical flow.
        gt_flow: ground truth optical flow.
        threshold: threshold for the error in the PEPN.
    Returns:
        msen: mean squared error.
        pepn: percentage of erroneous pixels.
    """

    diff_u = gt_flow[:,:,0] - pred_flow[:,:,0]
    diff_v = gt_flow[:,:,1] - pred_flow[:,:,1]

    sq_diff = np.sqrt(diff_u**2 + diff_v**2)
    sq_diff_valid = sq_diff[gt_flow[:,:,2] == 1]

    msen = np.mean(sq_diff_valid) 
    pepn = (np.sum(sq_diff_valid > threshold) / len(sq_diff_valid)) * 100

    #visualizations

    if plots:

        #Plot valid pixels
        plt.imshow(gt_flow[:,:,2])
        plt.title('Valid Pixels')
        plt.axis('off')
        plt.show()

        #plot the error flow
        plt.imshow(sq_diff)
        plt.title('Optical Flow error')
        plt.axis('off')
        plt.colorbar()
        plt.show()

        #plot the error histogram        
        plt.hist(sq_diff_valid, bins=100, density=True)
        plt.title('Error Histogram')
        plt.xlabel('Error')
        plt.ylabel('Pixels probablity')
        plt.axvline(msen, color='g', linestyle='dashed', label= "MSEN", linewidth=1)
        plt.legend(loc='upper right')
        plt.show()
    

    return msen, pepn

