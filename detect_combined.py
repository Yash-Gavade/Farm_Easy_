import os
import time
import RPi.GPIO as GPIO
import cv2
from ultralytics import YOLO

# ----- GPIO Pin Definitions (BCM numbering) -----
SER = 17      # 74HC595 SER pin
RCLK = 27     # 74HC595 RCLK (latch clock)
SRCLK = 22    # 74HC595 SRCLK (shift clock)
BUZZER_PIN = 18
MQ135_PIN = 23  # Digital output from MQ135 sensor

# ----- Initialize GPIO -----
GPIO.setmode(GPIO.BCM)
GPIO.setup(SER, GPIO.OUT)
GPIO.setup(RCLK, GPIO.OUT)
GPIO.setup(SRCLK, GPIO.OUT)
GPIO.setup(BUZZER_PIN, GPIO.OUT)
GPIO.setup(MQ135_PIN, GPIO.IN)

# Ensure buzzer off initially
GPIO.output(BUZZER_PIN, GPIO.LOW)

# ----- Function to shift out 8 bits to 74HC595 -----
def shift_out(data_byte):
    GPIO.output(RCLK, GPIO.LOW)
    for i in range(8):
        GPIO.output(SRCLK, GPIO.LOW)
        bit = (data_byte >> (7 - i)) & 1
        GPIO.output(SER, GPIO.HIGH if bit else GPIO.LOW)
        GPIO.output(SRCLK, GPIO.HIGH)
    GPIO.output(RCLK, GPIO.HIGH)

# ----- Function to turn ON LEDs and buzzer -----
def alert_on():
    # Example: Q0-Q3 HIGH turns on R, G, B LEDs + transistor base (enable cathode)
    shift_out(0b00001111)
    GPIO.output(BUZZER_PIN, GPIO.HIGH)
    print("[ALERT] LEDs and buzzer ON")

# ----- Function to turn OFF LEDs and buzzer -----
def alert_off():
    shift_out(0b00000000)
    GPIO.output(BUZZER_PIN, GPIO.LOW)
    print("[ALERT] LEDs and buzzer OFF")

# ----- Load YOLO model -----
MODEL_PATH = "/home/rpi/Farm_Easy_/Yolov10/models/best10n.pt"
print(f"Loading YOLO model from: {MODEL_PATH}")
model = YOLO(MODEL_PATH)
print("YOLO model loaded successfully.")

# ----- Initialize video capture (camera index = 1) -----
VIDEO_DEVICE_INDEX = 1
cap = cv2.VideoCapture(VIDEO_DEVICE_INDEX)
if not cap.isOpened():
    print(f"ERROR: Cannot open camera at index {VIDEO_DEVICE_INDEX} (/dev/video{VIDEO_DEVICE_INDEX})")
    exit(1)
else:
    print(f"Camera /dev/video{VIDEO_DEVICE_INDEX} opened successfully.")

try:
    while True:
        # Read MQ135 sensor digital value (e.g. gas detected or not)
        gas_detected = GPIO.input(MQ135_PIN)
        print(f"MQ135 sensor digital reading: {gas_detected}")

        # Capture frame from camera
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame from camera.")
            continue

        # Save captured frame temporarily
        timestamp = int(time.time())
        img_path = f"/home/rpi/Farm_Easy_/Yolov10/static/capture_{timestamp}.jpg"
        cv2.imwrite(img_path, frame)
        print(f"Saved captured frame to {img_path}")

        # Run YOLO detection on saved image
        results = model(img_path)
        # Save detection results on the same file (overwrites image)
        results[0].save(filename=img_path)
        print(f"YOLO detection completed and saved to {img_path}")

        # Example: check if "rotten banana" detected
        rotten_banana_detected = False
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                # Replace with your model's class id for rotten bananas
                if cls_id == 0:  # assuming class 0 = rotten banana
                    rotten_banana_detected = True
                    break
            if rotten_banana_detected:
                break

        # Logic to turn LEDs and buzzer on/off
        if rotten_banana_detected or gas_detected == 1:
            print("Rotten banana or gas detected!")
            alert_on()
        else:
            print("No rotten banana or gas detected.")
            alert_off()

        # Wait before next loop
        time.sleep(3)

except KeyboardInterrupt:
    print("Interrupted by user. Cleaning up...")

finally:
    alert_off()
    cap.release()
    GPIO.cleanup()
    print("Resources released. Exiting.")