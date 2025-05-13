import threading
import cv2
import numpy as np
import pyttsx3
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from tensorflow.keras.models import load_model
from ultralytics import YOLO
import base64

# Initialize FastAPI app
app = FastAPI()

# Load YOLO model (trained to detect fruits)
yolo_model = YOLO("D:/DOWNLOADS/BRAVE/Detection/Yolov10/yolov10n.pt")

# Load fruit freshness classifier
classifier = load_model("D:/DOWNLOADS/BRAVE/Detection/Yolov10/fruit_freshness_model.h5", compile=False)

# Text-to-speech setup
engine = pyttsx3.init()
engine.setProperty('rate', 150)
last_spoken = None

def speak_in_thread(text):
    def speak():
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=speak).start()

# Classify cropped fruit
def classify_fruit(img):
    try:
        resized = cv2.resize(img, (224, 224))
        normalized = resized / 255.0
        prediction = classifier.predict(np.expand_dims(normalized, axis=0), verbose=0)[0][0]
        return "Fresh" if prediction < 0.5 else "Spoiled"
    except:
        return "Unknown"

# Video frame generator
def gen_frames():
    global last_spoken
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Webcam not accessible")

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = yolo_model(frame)[0]

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            freshness = classify_fruit(crop)

            class_id = int(box.cls[0])
            class_name = yolo_model.names[class_id] if class_id in yolo_model.names else "Fruit"

            label = f"{class_name} - {freshness}"
            color = (0, 255, 0) if freshness == "Fresh" else (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if freshness == "Spoiled" and last_spoken != "Spoiled":
                speak_in_thread(f"Spoiled {class_name} detected")
                last_spoken = "Spoiled"
            elif freshness == "Fresh":
                last_spoken = None

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# === Web Interface ===
@app.get("/", response_class=HTMLResponse)
async def index():
    return """
    <html>
    <head>
        <title>Fruit Freshness Detection</title>
        <style>
            body { font-family: Arial; text-align: center; background: #f0f0f0; }
            h2 { color: #2e7d32; margin-top: 20px; }
            img { border: 4px solid #4caf50; margin-top: 20px; box-shadow: 0 0 10px rgba(0,0,0,0.3); }
        </style>
    </head>
    <body>
        <h2>Live Fruit Freshness Detection</h2>
        <img src="/video" width="720" height="540">
    </body>
    </html>
    """

@app.get("/video")
def video_feed():
    return StreamingResponse(gen_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

# Endpoint for analyzing base64 image (e.g. from mobile app)
@app.post("/analyze")
async def analyze_image(payload: dict):
    try:
        image_data = payload["image"].split(",")[1]
        img_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        results = yolo_model(img)[0]
        if not results.boxes:
            return JSONResponse({"result": "No fruit detected"})

        x1, y1, x2, y2 = map(int, results.boxes[0].xyxy[0])
        crop = img[y1:y2, x1:x2]
        freshness = classify_fruit(crop)

        class_id = int(results.boxes[0].cls[0])
        class_name = yolo_model.names[class_id] if class_id in yolo_model.names else "Fruit"

        label = f"{class_name} - {freshness}"
        return JSONResponse({"result": label})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# Run app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
