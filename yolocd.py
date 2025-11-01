from ultralytics import YOLO

model = YOLO("yolov8n-cls.pt")  # or yolov8s-cls.pt etc.
model.train(data="cls_dataset", epochs=50, imgsz=224, batch=16)
