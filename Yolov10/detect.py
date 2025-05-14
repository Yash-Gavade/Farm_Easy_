import os
import shutil
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import cv2
import time
from ultralytics import YOLO

# Initialize FastAPI
app = FastAPI()
cap = cv2.VideoCapture(0)

# Static and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load YOLOv10 model
model = YOLO("models/best10n.pt")  # Update the path if needed

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Streaming frames from webcam
def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

# Capture + YOLO detect
@app.post("/capture")
def capture():
    success, frame = cap.read()
    if not success:
        return JSONResponse(content={"error": "Failed to capture frame"}, status_code=500)

    # Save image
    timestamp = int(time.time())
    image_path = f"static/capture_{timestamp}.jpg"
    cv2.imwrite(image_path, frame)

    # Run YOLOv10 detection
    results = model(image_path)
    results[0].save(filename=image_path)  # Save output back to same path

    return {"img_path": f"/{image_path}"}

# Save captured image
@app.post("/save_result")
def save_result(img_path: str):
    try:
        # Extract filename from the path
        filename = img_path.split("/")[-1]
        save_dir = "saved_results"  # Create a directory to save the images

        # Check if the directory exists, if not create it
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Ensure the source image exists
        source_path = os.path.join("static", filename)
        if not os.path.exists(source_path):
            return JSONResponse(content={"error": "Captured image not found."}, status_code=404)

        # Define the destination path for saving the image
        new_path = os.path.join(save_dir, filename)
        
        # Copy the image to the 'saved_results' folder
        shutil.copy(source_path, new_path)

        return JSONResponse(content={"message": f"Image saved as {new_path}"}, status_code=200)
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    
