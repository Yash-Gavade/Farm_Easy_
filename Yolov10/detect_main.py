import os
import shutil
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import cv2
import time
from ultralytics import YOLO

print("Starting application...")

# Initialize FastAPI
app = FastAPI()

# Camera device index (use /dev/video1)
video_device_index = 1
print(f"Attempting to open camera at index {video_device_index} (/dev/video{video_device_index})")

# Open global camera once
cap = cv2.VideoCapture(video_device_index)
ret, frame = cap.read()
if ret:
    print(f"Camera /dev/video{video_device_index} works!")
else:
    print(f"Failed to open /dev/video{video_device_index}. Please check your camera connection.")

# Mount static files directory (static content served under /static)
static_dir = "Yolov10/static"
print(f"Mounting static files from directory: {static_dir}")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Templates directory
templates_dir = "templates"
print(f"Loading templates from directory: {templates_dir}")
templates = Jinja2Templates(directory=templates_dir)

# Load YOLOv10 model
model_path = "/home/rpi/Farm_Easy_/Yolov10/models/best10n.pt"
print(f"Loading YOLO model from: {model_path}")
try:
    model = YOLO(model_path)
    print("YOLO model loaded successfully.")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    raise e  # stop app if model fails to load

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    print("Serving home page '/'")
    return templates.TemplateResponse("index.html", {"request": request})

def generate_frames():
    print("Starting video frame generator")
    if not cap.isOpened():
        print(f"Failed to open camera at index {video_device_index} in generate_frames()")
        return
    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("Failed to read frame from camera in generate_frames()")
                break
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    except Exception as e:
        print(f"Exception in generate_frames(): {e}")

@app.get("/video_feed")
def video_feed():
    print("Client requested /video_feed")
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.post("/capture")
def capture():
    print("Capture endpoint called")
    if not cap.isOpened():
        print("Camera is not opened in capture()!")
        return JSONResponse(content={"error": "Camera is not available"}, status_code=500)

    success, frame = cap.read()
    if not success:
        print("Failed to read frame in capture()")
        return JSONResponse(content={"error": "Failed to capture frame"}, status_code=500)

    timestamp = int(time.time())
    image_path = f"{static_dir}/capture_{timestamp}.jpg"
    print(f"Saving captured frame to {image_path}")
    cv2.imwrite(image_path, frame)

    try:
        print(f"Running YOLO detection on {image_path}")
        results = model(image_path)
        results[0].save(filename=image_path)  # overwrite with detection output
        print(f"Detection results saved to {image_path}")
    except Exception as e:
        print(f"Error during YOLO detection: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

    # Return path relative to /static for client access
    relative_path = f"/static/capture_{timestamp}.jpg"
    return {"img_path": relative_path}

@app.post("/save_result")
def save_result(img_path: str):
    print(f"Save result called with img_path: {img_path}")
    try:
        filename = img_path.split("/")[-1]
        save_dir = "saved_results"
        print(f"Ensuring save directory exists: {save_dir}")

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print(f"Created directory: {save_dir}")

        source_path = os.path.join(static_dir, filename)
        if not os.path.exists(source_path):
            error_msg = f"Captured image not found at {source_path}"
            print(error_msg)
            return JSONResponse(content={"error": error_msg}, status_code=404)

        new_path = os.path.join(save_dir, filename)
        shutil.copy(source_path, new_path)
        print(f"Copied image from {source_path} to {new_path}")

        return JSONResponse(content={"message": f"Image saved as {new_path}"}, status_code=200)

    except Exception as e:
        print(f"Error in save_result endpoint: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
