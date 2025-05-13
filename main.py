from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import cv2
import os
import time
from ultralytics import YOLO

app = FastAPI()

# Mount directories
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load model
model = YOLO("D:/DOWNLOADS/BRAVE/Detection/Yolov10/best.pt")

@app.get("/", response_class=HTMLResponse)
def webcam_page(request: Request):
    return templates.TemplateResponse("webcam.html", {"request": request})

@app.post("/capture")
def capture_image():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    img_filename = f"static/captured_{int(time.time())}.jpg"

    if ret:
        cv2.imwrite(img_filename, frame)
        results = model(img_filename)
        results[0].save(filename=img_filename)  # overwrite with detection

    cap.release()
    cv2.destroyAllWindows()

    return RedirectResponse(url=f"/result?img={os.path.basename(img_filename)}", status_code=303)

@app.get("/result", response_class=HTMLResponse)
def result(request: Request, img: str):
    return templates.TemplateResponse("result.html", {"request": request, "img_path": f"static/{img}"})