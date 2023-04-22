import itertools
import pandas as pd
import os
import pickle

def load_from_txt(path):
    """
    :param path: path file

    :return: list = [[class,x1, y1, x2, y2]]
    """
    detections = []
    with open(path) as f:
        lines = f.readlines()

    for l in lines:
        ll = l.split(",")
        frame = int(ll[0]) 
        detections.append(
            [
                frame,
                0,
                float(ll[2]),
                float(ll[3]),
                float(ll[2]) + float(ll[4]),
                float(ll[3]) + float(ll[5]),
            ]
        )

    """Group the detected boxes by frame_id as a dictionary"""
    detections.sort(key=lambda x: x[0])
    detections = itertools.groupby(detections, key=lambda x: x[0])
    final_dict = {}
    for k,v in detections:
        det = []
        for vv in v:
            det.append(list(vv)[1:])
        final_dict[k] = det

    return final_dict


def write_csv(detections, out_path):
    df_list = []
    for frame_id in detections:
        for track in detections[frame_id]:
            width = track[3] - track[1]
            height = track[4] - track[2]
            bb_left = track[1]
            bb_top = track[2]
            df_list.append(
                pd.DataFrame({'frame': int(frame_id), 'id': int(track[-1]), 'bb_left': bb_left, 'bb_top': bb_top,
                              'bb_width': width, 'bb_height': height, 'conf': track[-2], "x": -1, "y": -1,
                              "z": -1}, index=[0]))

    df = pd.concat(df_list, ignore_index=True)
    df = df.sort_values(by=['frame'])
        
    # save the csv file without the header
    df.to_csv(out_path, index=False,header=False)

def load_pkl_file(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def read_all_pkl_files(folder_path):
    pkl_files = [f for f in os.listdir(folder_path) if f.endswith('.pkl')]

    all_data = {}
    for file_name in pkl_files:
        file_path = os.path.join(folder_path, file_name)
        data = load_pkl_file(file_path)
        all_data[file_name] = data

    return all_data

def convert_pkl_to_txt(pkl_folder, txt_folder):
    all_data = read_all_pkl_files(pkl_folder)

    for file_name, data in all_data.items():
        fname = file_name.split('.')[0]
        print(f"Writing content of {fname}: in txt format for TrackEval")

        # delete the first line of the csv file and save it to txt
        write_csv(data, f"{txt_folder}/{fname}.txt")
