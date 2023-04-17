from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Use the model
model.train(data="/export/home/group03/mcv-m6-2023-team6/week5/configs/finetuning_S01_S04_yolov8.yaml",epochs=20, device=0, imgsz=640, workers=2, val=False)  # train the model
metrics = model.val()  # evaluate model performance on the validation set
success = model.export(format="-")