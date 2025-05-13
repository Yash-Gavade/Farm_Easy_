import cv2
from ultralytics import YOLO
import time
import os

# Load the YOLOv10 trained model
model = YOLO("D:/DOWNLOADS/BRAVE/Detection/Yolov10/best.pt")

# Open webcam
cap = cv2.VideoCapture(0)
print("Press SPACE to capture image, ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Webcam - Press SPACE to capture", frame)

    key = cv2.waitKey(1)
    
    if key == 27:  # ESC to quit
        break
    elif key == 32:  # SPACE to capture
        timestamp = int(time.time())
        img_filename = f"captured_{timestamp}.jpg"
        cv2.imwrite(img_filename, frame)
        print(f"[INFO] Image saved: {img_filename}")

        # Run YOLOv10 inference
        results = model(img_filename)
        results[0].show()

        # Delete image after inference
        os.remove(img_filename)
        print(f"[INFO] Deleted: {img_filename}")

cap.release()
cv2.destroyAllWindows()
