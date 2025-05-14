import os
from ultralytics import YOLO

# === Load YOLOv8 trained model ===
model = YOLO("D:/DOWNLOADS/BRAVE/Detection/Yolov10/runs/detect/train/weights/best.pt")


# === Folder containing test images ===
image_folder = "D:/DOWNLOADS/BRAVE/Detection/Yolov10/test_fruits"
valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')

# === Loop through images in the folder ===
for image_name in os.listdir(image_folder):
    if not image_name.lower().endswith(valid_extensions):
        continue

    image_path = os.path.join(image_folder, image_name)
    
    # Run YOLO detection on the image
    results = model(image_path)[0]  # Perform detection on the image

    # Show the results (this will display each image one by one)
    results.show()  # This will open a window showing the annotated image
    
    # Wait for a key press to move to the next image
    print(f"Processed {image_name}")


