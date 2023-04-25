
# Welcome to the Week 5 report!

### In this report, we explore the Multi-Object Single Camera and Multi-Object Multi-Camera problems. For the Multi-Camera problem, we use an adapted version of this repository [vehicle_mtmc](https://github.com/regob/vehicle_mtmc)*.

**IMPORTANT**  
**This repository has been changed to support our tasks and it has been improved*.

## Run our code

### Before starting with the project, please ensure the following steps are completed:

```bash
export PYTHONPATH=$("./week5/vehicle_mtmc")
```

### If you want to run all experiments, use our job.

```bash
sbatch --gres gpu:1 -n 10 job
```

### End-to-End experiment

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


------------
## Yamls in this readme are examples. You can find our yaml's [here](https://github.com/mcv-m6-video/mcv-m6-2023-team6/tree/main/week5/vehicle_mtmc/config/AI_city).








