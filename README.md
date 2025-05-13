# 🍎 Fruit Freshness Detection using YOLOv8

This project is designed to detect the freshness of fruits using YOLOv8 for object detection and a Keras model for classification. It includes training, inference, and testing scripts, along with pre-trained models to assist with fruit freshness classification.

## 📁 Project Structure

```plaintext
DETECTION/
├── dataset/
│   ├── train/                 # Training images and labels
│   └── test/                  # Testing images and labels
├── models/
│   ├── best.pt               # Best YOLOv8 model from training
│   ├── yolov8n.pt            # YOLOv8 nano pre-trained model (used as base)
│   └── fruit_freshness_model.h5  # Keras model for classifying fruit freshness
├── runs/detect/
│   ├── train/                # YOLOv8 training results for different runs
│   └── val/                  # Validation result images (auto-generated)
├── Testing_model/
│   ├── test_fruits/          # Images for testing the final detection pipeline
│   └── detecting.py          # Script for running inference and displaying results
├── Training_model/           # (unused or empty folder; possibly for future use)
├── yolo_dataset/
│   ├── train/                # YOLOv8 training images
│   ├── val/                  # YOLOv8 validation images
│   └── data.yaml             # Dataset configuration file for YOLOv8
├── app.py                    # Optional main or app launcher file
├── best.torchscript          # TorchScript version of best.pt (for deployment)
├── fruit.h5                  # Duplicate or alternate Keras model
├── inference.py              # Standalone script for model inference
├── requirements.txt          # Python dependencies
└── Yolo_reorganize_dataset.ipynb # Notebook for organizing dataset structure
```



## 🚀 Getting Started

### 1. Install Dependencies  

Use the following command to install all required Python packages:  

pip install -r requirements.txt  




## How Training YOLOv8 Model was done   

To train your custom object detector using YOLOv8:  

yolo task=detect mode=train model=yolov8n.pt data="yolo_dataset/data.yaml" epochs=50 imgsz=224 batch=4  


## Testing / Inference  

### A. Using detecting.py   
Run the detection on test images using:  
python Testing_model/detecting.py

### B. Using inference.py  
Alternative inference logic (possibly for different model formats or structured outputs):  
python inference.py

## 📦 Model Formats  
best.pt: YOLOv8 PyTorch model saved after training.  

best.torchscript: YOLOv8 model converted to TorchScript for deployment.  

fruit_freshness_model.h5: A Keras model likely used for classifying freshness of detected fruit crops.

yolov8n.pt: YOLOv8 Nano base model used for fine-tuning.  

## 🧪 Testing Workflow
Detect fruits with the YOLO model using test images from Testing_model/test_fruits/.  

(Optional) Crop detected fruits and pass them into the Keras classifier (fruit_freshness_model.h5) for freshness classification.  

Display results (bounding boxes, class names, etc.).  

## The command:



uvicorn app:app --reload   
is used to run a FastAPI or Starlette web application. 
