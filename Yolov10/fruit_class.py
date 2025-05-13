from ultralytics import YOLO

# Load YOLOv10 pre-trained model
model = YOLO('yolov10n.pt')  # or yolov10s.pt for better accuracy

# Fine-tune on your custom dataset
model.train(
    data='Object_dataset/data.yaml',
    epochs=4,
    imgsz=640,
    batch=16,
    name='yolov10-bananas',
    device=0  # Use 'cpu' if no GPU
)
