import os
import cv2
from ultralytics import YOLO
import numpy as np

# === Load YOLOv8 trained model ===
model = YOLO("D:/DOWNLOADS/BRAVE/Detection/runs/detect/train6/weights/best.pt")

# === Folder containing test images ===
image_folder = "D:/DOWNLOADS/BRAVE/Detection/Yolov10/test_fruits"
valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')

# List to store annotated images
annotated_images = []

# === Loop through images ===
for image_name in os.listdir(image_folder):
    if not image_name.lower().endswith(valid_extensions):
        continue

    image_path = os.path.join(image_folder, image_name)
    image = cv2.imread(image_path)

    if image is None:
        print(f"❌ Error loading {image_name}")
        continue

    # === Run YOLO detection ===
    results = model(image)[0]

    # === Annotate results on image ===
    annotated = results.plot()

    # Append the annotated image to the list
    annotated_images.append(annotated)

    print(f"📷 {image_name}: {', '.join(results.names[int(cls)] for cls in results.boxes.cls)}")

# === Concatenate images horizontally ===
if annotated_images:
    # Resize all images to the same height (optional, but helps in concatenating)
    height = min(img.shape[0] for img in annotated_images)
    resized_images = [cv2.resize(img, (int(img.shape[1] * height / img.shape[0]), height)) for img in annotated_images]

    # Concatenate horizontally (side by side)
    combined_image = np.hstack(resized_images)

    # === Show the combined image with annotations ===
    cv2.imshow("Combined Results", combined_image)
    cv2.waitKey(0)  # Wait until a key is pressed
    cv2.destroyAllWindows()
