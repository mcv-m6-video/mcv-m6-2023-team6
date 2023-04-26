import os
from util import load_from_txt_video
import argparse
import cv2
import numpy as np
from skimage import io
from tqdm import tqdm

def load_motchallenge_format(file_path, frame_offset=1):
    """Loads a MOTChallenge annotation txt, with frame_offset being the index of the first frame of the video"""
    res = []
    with open(file_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split(",")
            line = [int(x) for x in line[:6]] + [float(x) for x in line[6:]]

            # Subtract frame offset from frame indices. If indexing starts at one, we convert
            # it to start from zero.
            line[0] -= frame_offset
            res.append(tuple(line))
    return detection_list_to_dict(res)


colors =  {1:(255,0,0),2:(0,255,0),3:(0,0,255),4:(255, 255, 0),5: (128, 0, 128),6:(0, 255, 255),7:(255, 0, 255),8:(255, 165, 0),9:(255, 20, 147),10:(165, 42, 42),11:(0, 128, 128),12:(75, 0, 130),
            13:(238, 130, 238),14:(128, 128, 0),15:(128, 0, 0),16:(255, 215, 0),17:(192, 192, 192),18:(0, 0, 128),19:(0, 255, 255),20:(255, 127, 80),21:(0, 255, 0),22:(255, 0, 255),23:(64, 224, 208),24:(245, 245, 220),25: (221, 160, 221)} 

colors = np.random.uniform(0, 255, size=(25, 3))

offsets = {10: 8.715, 11: 8.457, 12: 5.879, 13: 0.0, 14: 5.042, 15: 8.492}

def video(args,fps=10.0):
    if args.cam == 'c015':
        fps = 8.0
        
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_out = cv2.VideoWriter(f"/ghome/group03/mcv-m6-2023-team6/week5/Results/{args.output}/{args.seq}_{args.cam}.mp4", fourcc, fps, (1920, 1080))

    tracks = load_from_txt_video(args.tracking)
    tracks_ids = [det[0][0] for det in tracks.values()]
    
    
    #get the frames in order
    frames = os.listdir(f'/ghome/group03/dataset/aic19-track1-mtmc-train/train/{args.seq}/{args.cam}/frames')
    frames = sorted(frames, key=lambda x: int(x.split('.')[0]))
    tracks_offsets = load_motchallenge_format(frames, frame_offset=offsets[args.cam])
    
    # current path file
    for frame in tqdm(frames):
        frames_path = f'/ghome/group03/dataset/aic19-track1-mtmc-train/train/{args.seq}/{args.cam}/frames/{frame}'
        im = io.imread(frames_path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        id = int(frame.split('.')[0])

        if args.from_repo:
            id = id -1

        c = 1
        if id in tracks.keys():
            frame_boxes = tracks[id]
            for box in frame_boxes:
                track_id = box[0]

                if track_id not in colors:
                    color = colors[c]
                    c += 1

                    if c == 24:
                        c = 1
                else:
                    color = colors[track_id]

                cv2.rectangle(im, (int(box[1]), int(box[2])), (int(box[3]), int(box[4])), color, 2)
                cv2.putText(im, str(track_id), (int(box[1]), int(box[2])), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2,
                            cv2.LINE_AA)

        video_out.write(im)
    video_out.release()




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Video generation')
    parser.add_argument('--dataset_path', type=str, default='/export/home/group03/dataset/aic19-track1-mtmc-train/train/', help='Dataset directory')
    parser.add_argument('--output', type=str, required=True, help='Folder to save results')
    parser.add_argument('--seq', type=str, required=True, help='Sequence to use')
    parser.add_argument('--cam', type=str, required=True, help='Camera within the sequence')
    parser.add_argument('--tracking', type=str, required=True, help='Text files with tracking results')
    parser.add_argument('--from_repo', type=str, required=False, default=None)

    args = parser.parse_args()

    """ if not os.path.exists(f'/ghome/group03/mcv-m6-2023-team6/week5/Results/{args.output}'):
        os.makedirs(f'/ghome/group03/mcv-m6-2023-team6/week5/Results/{args.output}') """

    video(args)
