import itertools

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

