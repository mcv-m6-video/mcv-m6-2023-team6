import os
from collections import defaultdict

import xmltodict

""" def load_from_xml(path):

    frame_dict = defaultdict(list)
    for event, elem in ET.iterparse(path, events=('start',)):
        if elem.tag == 'track' and elem.attrib.get('label') == 'car':
            for x in (child.attrib for child in elem):
                frame = f"f_{x['frame']}"
                frame_dict[frame].append([float(x['xtl']), float(x['ytl']),
                                          float(x['xbr']), float(x['ybr'])])
    return frame_dict """


def load_from_xml(path):
    """

    :param path: path file

    :return: dict[frame_num] = [[x1, y1, x2, y2]]
    """

    with open(path) as f:
        tracks = xmltodict.parse(f.read())["annotations"]["track"]

    gt = defaultdict(list)
    num_iter = 0
    for track in tracks:
        label = track["@label"]
        boxes = track["box"]
        for box in boxes:
            if label == "car":
                frame = int(box["@frame"])
                frame = f"f_{frame}"
                gt[frame].append(
                    [
                        float(box["@xtl"]),
                        float(box["@ytl"]),
                        float(box["@xbr"]),
                        float(box["@ybr"]),
                    ]
                )
                num_iter += 1

            else:
                continue

    return gt, num_iter


def load_from_txt(path):
    """
    :param path: path file

    :return: list = [[frame,x1, y1, x2, y2, conf]]
    """
    frame_list = []
    with open(path) as f:
        lines = f.readlines()

    for l in lines:
        ll = l.split(",")
        frame = f"f_{int(ll[0]) - 1}"
        frame_list.append(
            [
                frame,
                float(ll[2]),
                float(ll[3]),
                float(ll[2]) + float(ll[4]),
                float(ll[3]) + float(ll[5]),
                ll[6],
            ]
        )

    return frame_list


if __name__ == "__main__":
    # Set the parent directory of your current directory
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), "../.."))

    # Set the relative path to the XML file
    relative_path = "dataset/ai_challenge_s03_c010-full_annotation.xml"

    # Get the absolute path of the XML file
    path = os.path.join(parent_dir, relative_path)

    # Print the absolute path
    print(path)
    frame_dict = load_from_xml(path)
    print(frame_dict)
