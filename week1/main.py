import argparse
import os

import yaml

from utils.rendering import rendering_video
from utils.utils import load_from_xml


def main(cfg):
    os.makedirs(f"runs/{cfg['run_name']}/", exist_ok=True)
    print(f"Run Name: {cfg['run_name']}")
    print(f"Run Mode: {cfg['run_mode']}")

    gt_boxes, total = load_from_xml(cfg['paths']['annotations_path'])
    rendering_video(f"runs/{cfg['run_name']}/video.mp4", gt_boxes, total, cfg['detection'][args.run_mode],
                    cfg['paths']['video_path'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--run_mode', required=True, type=str, help='Yolo, RCNN or SSD')
    parser.add_argument('-n', '--run_name', required=True, type=str, help='Run Folder Name')
    parser.add_argument("-c", '--config', default="configs/config.yml")
    args = parser.parse_args()

    # get the path of this file
    path = os.path.dirname(os.path.realpath(__file__))
    path_config = os.path.join(path, args.config)

    with open(path_config, "r") as f:
        config = yaml.safe_load(f)

    config['run_mode'] = args.run_mode
    config['run_name'] = args.run_name

    main(config)
