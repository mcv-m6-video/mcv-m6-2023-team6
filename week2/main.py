import argparse
import os

import yaml

from models import Gaussian, AdaptiveGaussian, SOTA
from utils.rendering import rendering_video

TOTAL_FRAMES_VIDEO = 2141


def main(cfg):
    os.makedirs(f"runs/{cfg['run_name']}/", exist_ok=True)
    print(f"Run Name: {cfg['run_name']}")
    print("----------------------------------------")

    frames_modelling = int(TOTAL_FRAMES_VIDEO * cfg["percentatge"])

    path = os.path.dirname(os.path.abspath(__file__))

    if cfg["run_mode"] == "Gaussian":
        print("Gaussian Function")
        print("----------------------------------------")
        model = Gaussian(cfg['paths']['video_path'], frames_modelling, alpha=cfg['alpha'], colorspace='gray',
                         checkpoint=f"{cfg['colorspace']}_{cfg['percentatge']}")

    elif cfg["run_mode"] == "AdaptativeGaussian":
        model = AdaptiveGaussian(cfg['paths']['video_path'], frames_modelling, p=cfg["rho"], alpha=cfg['alpha'],
                                 colorspace=cfg['colorspace'], checkpoint=f"{cfg['colorspace']}_{cfg['percentatge']}")

    elif cfg["run_mode"] == "SOTA":
        model = SOTA(cfg['paths']['video_path'], frames_modelling, p=cfg["rho"], checkpoint=None, n_jobs=-1,
                     method='MOG')
    else:
        raise ValueError("Invalid run mode")

    rendering_video(cfg, model, frames_modelling, './week2/results/task1.1/', cfg['paths']['annotations_path'])

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
    parser.add_argument("-p", "--percentatge", required=True, default=False, type=float,
                        help="Percentatge of video to use background")
    parser.add_argument("-e", "--sota_method", default="MOG", type=str,
                        help="SOTA method to use (MOG, MOG2, LSBP, ViBE")
    parser.add_argument("-a", "--alpha", default=0.25, type=float, help="Alpha Thresholding")
    parser.add_argument("--rho", default=0.05, type=float, help="Rho Thresholding")
    parser.add_argument("--colorspace", default="gray", type=str, help="Colorspace to use (gray, rgb, hsv, yuv)")

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
    config["percentatge"] = args.percentatge
    config["sota_method"] = args.sota_method
    config["alpha"] = args.alpha
    config["rho"] = args.rho
    config["colorspace"] = args.colorspace

    main(config)
