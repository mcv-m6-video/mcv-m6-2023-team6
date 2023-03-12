import argparse
import os

import yaml


def main(cfg):
    os.makedirs(f'runs/{cfg.run_name}/', exist_ok=True)

    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', required=True, type=str, help='Yolo, RCNN or SSD')
    parser.add_argument('-n', '--name', required=True, type=str, help='Run Folder Name')
    parser.add_argument("-c", '--configs', default="configs/config.yml")
    args = parser.parse_args()

    # get the path of this file
    path = os.path.dirname(os.path.realpath(__file__))
    path_config = os.path.join(path, args.config)

    with open(path_config, "r") as f:
        config = yaml.safe_load(f)
    main(config)
