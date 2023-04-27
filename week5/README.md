
# Welcome to the Week 5 report!

In this report, we explore the Multi-Object Single Camera and Multi-Object Multi-Camera problems. For the Multi-Camera problem, we use an adapted version of this repository [vehicle_mtmc](https://github.com/regob/vehicle_mtmc)*.

**IMPORTANT**  
**This repository has been changed to support our tasks*.

Before starting with the project, please ensure the following steps are completed:

```bash
export PYTHONPATH=$("./week5/vehicle_mtmc")
```

## Run all experiments in one command:

For SLURM:
```bash
sbatch --gres gpu:1 -n 10 job
```
For bash:  
```bash
bash job
```


## Run the experiments using Maximum Iou Overlap tracking algorithm step by step:
#### Detections
Inference with [Yolov8](https://github.com/ultralytics/ultralytics) to get the detections
```
python inference_yolov8.py
```

#### Tracking 
Tracking using the Max Iou Overlap (MIO) alogrithm developed in Week3: 
```bash
python pre_MTMC.py --OF 0
```
Tracking using the Max Iou Overlap with Optical Flow (MIO-OF) developed in Week4:
```bash
python pre_MTMC.py --OF 1
```

#### Feature extraction
Extract Features from detected bounding boxes using Resnet50:
For MIO:
```bash
python pre_MTMC.py --OF 0
```
For MIO-OF
```bash
python pre_MTMC.py --OF 1
```

#### Re-Id model (MTMC)
For MIO:
```bash
python vehicle_mtmc/mtmc/run_mtmc.py --config AI_city/mtmc_s01_max_iou.yaml
```
```bash
python vehicle_mtmc/mtmc/run_mtmc.py --config AI_city/mtmc_s03_max_iou.yaml
```
```bash
python vehicle_mtmc/mtmc/run_mtmc.py --config AI_city/mtmc_s04_max_iou.yaml
```
For MIO-OF
```bash
python vehicle_mtmc/mtmc/run_mtmc.py --config AI_city/mtmc_s01_max_iou_OF.yaml
```
```bash
python vehicle_mtmc/mtmc/run_mtmc.py --config AI_city/mtmc_s03_max_iou_OF.yaml
```
```bash
python vehicle_mtmc/mtmc/run_mtmc.py --config AI_city/mtmc_s04_max_iou_OF.yaml
```

### Run the experiments using DeepSort and ByteTrack as trackers

```bash
python mtmc/run_express_mtmc.py --config AI_city/end2end_ByTrack_s01.yaml
```

### Run Tracker (DeepSort or Bytrack)
```bash
python3 mot/run_tracker.py --config AI_city/mot.yaml
```

### Run MTMC

```bash
python mtmc/mtmc.py --config AI_city/express_s01.yaml
```



## Run Evaluation with TrackEval for individual cameras
Note that the ground truth must be in the appropriate format and the directories as well, defined in the TrackEval library. You can find the documentation for the
MOTChallenge and it's format [here](https://github.com/JonathonLuiten/TrackEval/tree/master/docs/MOTChallenge-Official). Once the directories are correctly created, run the evaluation with:
```bash
python run_mot_challenge.py --DO_PREPROC False 
```

------------
## You can find our yaml's [here](https://github.com/mcv-m6-video/mcv-m6-2023-team6/tree/main/week5/vehicle_mtmc/config/AI_city).

--------
## You can find HyperParameters explanations [here](https://github.com/mcv-m6-video/mcv-m6-2023-team6/tree/main/week5/vehicle_mtmc/config/defaults.py).







