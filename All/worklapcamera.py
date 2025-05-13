import threading
import cv2
import numpy as np
import pyttsx3
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse
from tensorflow.keras.models import load_model
from ultralytics import YOLO

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

# Generator to yield frames from mobile camera
def gen_frames():
    global last_spoken
    cap = cv2.VideoCapture(0)  # Open the mobile camera (can work via mobile browser)
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

# HTML page for mobile camera
@app.get("/", response_class=HTMLResponse)
async def index():
    return """
    <html>
    <head>
        <title>Fruit Freshness Detection (Real-time)</title>
    </head>
    <body style="text-align:center; font-family:sans-serif;">
        <h2>Real-time Fruit Freshness Detection (Mobile Camera)</h2>
        <img src="/video" id="video" width="720" height="540">
        <p>Fresh or spoiled? Let’s find out live!</p>
    </body>
    </html>
    """

# Video streaming route (for mobile camera)
@app.get("/video")
def video_feed():
    return StreamingResponse(gen_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

# Run the FastAPI server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
