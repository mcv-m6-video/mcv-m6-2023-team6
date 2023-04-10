# Week 4

### Task 1: Optical Flow

#### Task 1.1: Block Matching
   + Exhauastive search (SSD, SAD)
   + Template matching (NCC)
   + Grid search to optimize hyperparameters: motion type (forward, backward), block size (8, 16, .., 128) and search area (8, 16, .., 128)

#### Task 1.2: Off-the-shelf
Clone the following repos to use each algorithm:
   + [Pyflow](https://github.com/pathak22/pyflow)
   + [Lucas Kanade](https://docs.opencv.org/3.3.1/dc/d6b/group__video__track.html#ga473e4b886d0bcc6b65831eb88ed93323)
   + [MaskFlownet](https://github.com/microsoft/MaskFlownet)
   + [RAFT](https://github.com/princeton-vl/RAFT)
   + [LiteFlowNet](https://github.com/sniklaus/pytorch-liteflownet)
   + [DEQ-Flow](https://github.com/locuslab/deq-flow)
```bash
python task_1_2.py
```
   + [Perceiver-IO](https://github.com/krasserm/perceiver-io.git)

Follow the instructions of the original repo to create perceiver-io environment. Then with this environment, run the following command:
```bash
python task_1_2_perceiver.py
```

#### Task 1.3: Improve tracking with optical flow


### Task 2: Multi Target Single Camera Tracking
For AI City Challenge
   + With and without optical flow

