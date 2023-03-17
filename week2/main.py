import argparse
import os

import yaml
from utils.rendering import rendering_video
from utils.util import load_from_txt, load_from_xml
from utils.metrics import generate_noisy_boxes

TOTAL_FRAMES = 2141


def main(cfg):
    os.makedirs(f"runs/{cfg['run_name']}/", exist_ok=True)
    print(f"Run Name: {cfg['run_name']}")
    print("----------------------------------------")

    frames_modelling = int(TOTAL_FRAMES * cfg["percentatge"])

    if cfg["run_mode"] == "GaussianModel":
        print("Gaussian Function")
        print("----------------------------------------")
        model = GaussianModel(cfg)

    elif cfg["run_mode"] == "AdaptativeGaussianModel":
        model = AdaptativeGaussianModel(cfg)

    else:
        raise ValueError("Invalid run mode")

    print("Done!")
    print("----------------------------------------")



if __name__ == "__main__":
    # check ffmepg in your system

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--run_mode", required=True, type=str, help="Gaussian Modelling")
    parser.add_argument("-r", "--run_name", required=True, type=str, help="Run Folder Name")
    parser.add_argument("-c", "--config", default="configs/config.yml")
    parser.add_argument("-s", "--save", default=True, type=bool, help="Save the video or not")
    parser.add_argument("-d", "--display", default=False, type=bool, help="Show the video or not")
    parser.add_argument("-p", "--percentatge", default=False, type=bool, help="Percentatge of video to use background")
    args = parser.parse_args()

    # get the path of this file
    path = os.path.dirname(os.path.realpath(__file__))
    path_config = os.path.join(path, args.config)

    with open(path_config) as f:
        config = yaml.safe_load(f)

    config["run_mode"] = args.run_mode
    config["run_name"] = args.run_name
    config["save"] = args.save
    config["display"] = args.display
    config["noisy"] = args.noisy

    main(config)
