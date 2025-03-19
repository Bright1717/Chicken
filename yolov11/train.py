from ultralytics import YOLO

model = YOLO("yolo11n.pt")


train_results = model.train(
    data="D:/yolov11/dataset/data.yaml",  # path to dataset YAML
    epochs=10,  # number of training epochs
    imgsz=640,  # training image size
    device="0",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
    workers = 0
)

metrics = model.val()