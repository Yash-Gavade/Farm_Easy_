from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from ultralytics import YOLO
import pyttsx3
import threading

# Initialize FastAPI app
app = FastAPI()

# Load models
# yolo_model = YOLO("D:/DOWNLOADS/BRAVE/Detection/runs/detect/train6/weights/best.pt") 
yolo_model = YOLO("models/yolov8n.pt")
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

# Generator to yield frames
def gen_frames():
    global last_spoken
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Webcam not available")

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

# UI page
@app.get("/", response_class=HTMLResponse)
async def index():
    return """
    <html>
    <head>
        <title>Fruit Freshness Detector</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                text-align: center;
                background: #f0f0f0;
                margin: 0;
                padding: 0;
            }
            h2 {
                color: #2e7d32;
                margin-top: 20px;
            }
            #video {
                border: 4px solid #4caf50;
                margin-top: 20px;
                box-shadow: 0 0 10px rgba(0,0,0,0.3);
            }
        </style>
    </head>
    <body>
        <h2>Live Fruit Freshness Detection</h2>
        <img src="/video" id="video" width="720" height="540">
        <p>Fresh or spoiled? Let’s find out live!</p>
    </body>
    </html>
    """

# Video streaming route
@app.get("/video")
def video_feed():
    return StreamingResponse(gen_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)
