import os
import subprocess
import sys
import urllib.request
import zipfile
import platform
import threading
import cv2
import numpy as np
import pyttsx3
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse
from tensorflow.keras.models import load_model
from ultralytics import YOLO

# Function to download ngrok
def download_ngrok():
    # Determine the platform
    system_platform = platform.system().lower()
    
    if system_platform == 'windows':
        ngrok_url = 'https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-windows-amd64.zip'
        ngrok_filename = 'ngrok-stable-windows-amd64.zip'
    elif system_platform == 'darwin':
        ngrok_url = 'https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-darwin-amd64.zip'
        ngrok_filename = 'ngrok-stable-darwin-amd64.zip'
    elif system_platform == 'linux':
        ngrok_url = 'https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip'
        ngrok_filename = 'ngrok-stable-linux-amd64.zip'
    else:
        raise Exception(f"Unsupported OS: {system_platform}")

    # Download ngrok
    print("Downloading ngrok...")
    urllib.request.urlretrieve(ngrok_url, ngrok_filename)

    # Unzip ngrok
    with zipfile.ZipFile(ngrok_filename, 'r') as zip_ref:
        zip_ref.extractall()

    print("ngrok downloaded and extracted.")
    os.remove(ngrok_filename)

# Function to run ngrok as a subprocess
def run_ngrok():
    print("Starting ngrok...")
    ngrok_process = subprocess.Popen(["ngrok", "http", "8000"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return ngrok_process

# Initialize FastAPI app
app = FastAPI()

# Load models
yolo_model = YOLO("models/yolov8n.pt")  # Ensure you have YOLOv8 model file
classifier = load_model("models/fruit_freshness_model.h5", compile=False)

# TTS setup
engine = pyttsx3.init()
engine.setProperty('rate', 150)
last_spoken = None

# Run TTS in thread
def speak_in_thread(text):
    def speak():
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=speak).start()

# Classify a single cropped fruit image
def classify_fruit(img):
    try:
        resized = cv2.resize(img, (224, 224))
        normalized = resized / 255.0
        prediction = classifier.predict(np.expand_dims(normalized, axis=0), verbose=0)[0][0]
        return "Fresh" if prediction < 0.5 else "Spoiled"
    except:
        return "Unknown"

# Generator to yield frames from the mobile camera feed
def gen_frames():
    global last_spoken
    cap = cv2.VideoCapture(0)  # Use default webcam on server

    if not cap.isOpened():
        raise RuntimeError("Camera not available")

    while True:
        success, frame = cap.read()
        if not success:
            break

        # YOLO detection
        results = yolo_model(frame)[0]

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            label = classify_fruit(crop)
            color = (0, 255, 0) if label == "Fresh" else (0, 0, 255)

            # Draw label and box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Speak once per label change
            if label == "Spoiled" and last_spoken != "Spoiled":
                speak_in_thread("Spoiled fruit detected!")
                last_spoken = "Spoiled"
            elif label == "Fresh":
                last_spoken = None

        # Encode as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


# HTML page for mobile camera
@app.get("/", response_class=HTMLResponse)
async def index():
    return """
<html>
<head>
    <title>Fruit Freshness Detection (Real-time)</title>
    <style>
        body {
            font-family: sans-serif;
            text-align: center;
        }
        video {
            width: 100%;
            height: auto;
            border: 1px solid #ddd;
        }
    </style>
</head>
<body>
    <h2>Real-time Fruit Freshness Detection (Mobile Camera)</h2>
    <video id="video" autoplay playsinline></video>
    <p>Fresh or spoiled? Let’s find out live!</p>
    
<script>
    const video = document.getElementById('video');
    const videoStreamURL = '/video';  // This points to the video stream route in FastAPI

    // Request access to the device's camera
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({ video: true })
            .then(function (stream) {
                // Use the video stream from the user's camera and show it in the video element
                video.srcObject = stream;
            })
            .catch(function (error) {
                console.error('Error accessing the camera: ', error);
                alert('Unable to access camera. Please check permissions.');
            });
    } else {
        alert('Camera not supported by this browser.');
    }
</script>

</body>
</html>
    """

# Video streaming route (for mobile camera)
@app.get("/video")
def video_feed():
    return StreamingResponse(gen_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

# Run the FastAPI server and ngrok
if __name__ == "__main__":
    # Download ngrok if not available
    if not os.path.exists("ngrok.exe"):
        download_ngrok()

    # Run ngrok to expose the FastAPI server
    ngrok_process = run_ngrok()

    # Start FastAPI server
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
