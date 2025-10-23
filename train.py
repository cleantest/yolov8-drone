from ultralytics import YOLO

# Load a pretrained YOLOv8 model (tiny version for speed)
model = YOLO('yolov8n.pt')

# Train on your custom dataset
model.train(data='data.yaml', epochs=30, imgsz=416, batch=16)
