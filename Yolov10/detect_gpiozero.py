import os
import shutil
import time
import logging
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import cv2
from ultralytics import YOLO
from gpiozero import OutputDevice, DigitalInputDevice, Buzzer

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("FarmEasy")

app = FastAPI()

# ========== GPIO Setup using gpiozero ==========

# GPIO pins connected to the shift register outputs (Q0, Q1, Q2, Q3)
# Adjust these pins according to your wiring!
RED_LED_PIN = 17     # Q0 -> Red LED anode resistor
GREEN_LED_PIN = 27   # Q1 -> Green LED anode resistor
BLUE_LED_PIN = 22    # Q2 -> Blue LED anode resistor
TRANSISTOR_PIN = 23  # Q3 -> Base of PN2222 transistor controlling common cathode

# MQ135 digital output connected to this pin
MQ135_DO_PIN = 24    # Digital output pin from MQ135 sensor

# Buzzer connected to this pin
BUZZER_PIN = 18

logger.info("Setting up gpiozero devices")

# Setup LEDs via OutputDevice (assuming active HIGH to turn ON)
red_led = OutputDevice(RED_LED_PIN, active_high=True, initial_value=False)
green_led = OutputDevice(GREEN_LED_PIN, active_high=True, initial_value=False)
blue_led = OutputDevice(BLUE_LED_PIN, active_high=True, initial_value=False)

# Transistor to switch common cathode (active HIGH to enable)
transistor = OutputDevice(TRANSISTOR_PIN, active_high=True, initial_value=False)

# MQ135 digital input device
mq135_sensor = DigitalInputDevice(MQ135_DO_PIN, pull_up=False)  # Assume module output is open drain / active HIGH

# Buzzer device
buzzer = Buzzer(BUZZER_PIN)

# ========== Camera & Model Setup ==========

VIDEO_DEVICE_INDEX = 1  # /dev/video1 as you confirmed

logger.info(f"Trying to open camera at index {VIDEO_DEVICE_INDEX}")
cap = cv2.VideoCapture(VIDEO_DEVICE_INDEX)
if not cap.isOpened():
    logger.error(f"Cannot open camera at index {VIDEO_DEVICE_INDEX}. Check connection!")
else:
    logger.info(f"Camera at index {VIDEO_DEVICE_INDEX} opened successfully.")

# Mount static files and templates
STATIC_DIR = "Yolov10/static"
TEMPLATES_DIR = "templates"

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Load YOLO model
MODEL_PATH = "/home/rpi/Farm_Easy_/Yolov10/models/best10n.pt"
logger.info(f"Loading YOLO model from {MODEL_PATH}")
try:
    model = YOLO(MODEL_PATH)
    logger.info("YOLO model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load YOLO model: {e}")
    raise

# ========== Utility functions ==========

def turn_on_rgb(r: bool, g: bool, b: bool):
    red_led.value = r
    green_led.value = g
    blue_led.value = b
    transistor.value = r or g or b  # Enable transistor if any LED color is ON

def indicate_spoiled():
    # Show red LED and buzzer alert for spoilage
    logger.info("Indicating spoiled fruit: Red LED ON, buzzer ON")
    turn_on_rgb(True, False, False)
    buzzer.on()

def indicate_fresh():
    # Show green LED and turn buzzer off for fresh fruit
    logger.info("Indicating fresh fruit: Green LED ON, buzzer OFF")
    turn_on_rgb(False, True, False)
    buzzer.off()

def indicate_idle():
    # Turn off all LEDs and buzzer
    logger.info("Indicating idle state: All LEDs OFF, buzzer OFF")
    turn_on_rgb(False, False, False)
    buzzer.off()

# ========== FastAPI routes ==========

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    logger.info("Serving home page '/'")
    # Also pass sensor status for display if needed
    sensor_status = "High" if mq135_sensor.value else "Low"
    return templates.TemplateResponse("index.html", {"request": request, "sensor_status": sensor_status})

def generate_frames():
    logger.info("Starting video frame generator")
    cap_local = cv2.VideoCapture(VIDEO_DEVICE_INDEX)
    if not cap_local.isOpened():
        logger.error(f"Failed to open camera at index {VIDEO_DEVICE_INDEX} in generate_frames()")
        return

    try:
        while True:
            success, frame = cap_local.read()
            if not success:
                logger.error("Failed to read frame from camera in generate_frames()")
                break
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        cap_local.release()
        logger.info("Released camera resource in generate_frames()")

@app.get("/video_feed")
async def video_feed():
    logger.info("Client requested /video_feed")
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.post("/capture")
async def capture():
    logger.info("Capture endpoint called")

    cap_local = cv2.VideoCapture(VIDEO_DEVICE_INDEX)
    if not cap_local.isOpened():
        logger.error(f"Failed to open camera at index {VIDEO_DEVICE_INDEX} in capture()")
        return JSONResponse(content={"error": "Camera not accessible"}, status_code=500)

    try:
        success, frame = cap_local.read()
        if not success:
            logger.error("Failed to capture frame")
            return JSONResponse(content={"error": "Failed to capture frame"}, status_code=500)

        timestamp = int(time.time())
        image_path = f"{STATIC_DIR}/capture_{timestamp}.jpg"
        logger.info(f"Saving captured frame to {image_path}")
        cv2.imwrite(image_path, frame)

        logger.info(f"Running YOLO detection on {image_path}")
        results = model(image_path)
        results[0].save(filename=image_path)
        logger.info(f"Detection results saved to {image_path}")

        # Use MQ135 sensor to indicate spoilage or freshness (assuming digital HIGH means spoiled)
        if mq135_sensor.value:
            indicate_spoiled()
        else:
            indicate_fresh()

        return {"img_path": f"/static/capture_{timestamp}.jpg"}
    except Exception as e:
        logger.error(f"Error in capture endpoint: {e}")
        indicate_idle()
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        cap_local.release()
        logger.info("Released camera resource in capture()")

@app.post("/save_result")
async def save_result(img_path: str):
    logger.info(f"Save result called with img_path: {img_path}")
    try:
        filename = img_path.split("/")[-1]
        save_dir = "saved_results"

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            logger.info(f"Created directory: {save_dir}")

        source_path = os.path.join(STATIC_DIR, filename)
        if not os.path.exists(source_path):
            error_msg = f"Captured image not found at {source_path}"
            logger.error(error_msg)
            return JSONResponse(content={"error": error_msg}, status_code=404)

        new_path = os.path.join(save_dir, filename)
        shutil.copy(source_path, new_path)
        logger.info(f"Copied image from {source_path} to {new_path}")

        return JSONResponse(content={"message": f"Image saved as {new_path}"}, status_code=200)

    except Exception as e:
        logger.error(f"Error in save_result endpoint: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

# ========== Cleanup on shutdown ==========
@app.on_event("shutdown")
def cleanup():
    logger.info("Application shutdown: turning off LEDs and buzzer")
    indicate_idle()
    if cap.isOpened():
        cap.release()
        logger.info("Released global camera resource")