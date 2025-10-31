# YOLO Classification Web Interface

A simple Vue 2 + Flask web application for classifying office items using YOLO.

## 🚀 Quick Start

### Option 1: Use the batch file (Windows)
```bash
start.bat
```

### Option 2: Manual start
```bash
# Install requirements
pip install -r requirements.txt

# Start the server
python run.py
```

## 📁 Project Structure

```
ui-cls/
├── app.py          # Flask backend server
├── index.html      # Vue 2 frontend (single file)
├── app.js          # Vue 2 app logic
├── style.css       # Modern CSS styles
├── run.py          # Server launcher
├── start.bat       # Windows quick start
├── requirements.txt # Python dependencies
└── README.md       # This file
```

## 🎯 Features

- **Vue 2 Interface**: Reactive, modern UI with drag & drop
- **Image Upload**: Classify any image with confidence scores
- **Live Camera**: Real-time classification via OpenCV window
- **Top 5 Predictions**: Shows confidence bars for results
- **Responsive Design**: Works on desktop and mobile

## 📋 Requirements

- Python 3.8+
- Trained YOLO model at: `runs/classify/train2/weights/best.pt`
- Camera (optional, for live classification)

## 🔧 Model Training

Train your model first using:
```python
# yolocd.py
from ultralytics import YOLO
model = YOLO("yolov8n-cls.pt")
model.train(data="cls_dataset", epochs=50, imgsz=224, batch=16)
```

## 🌐 Usage

1. Start the server (see Quick Start above)
2. Open http://localhost:5001 in your browser
3. Upload an image or start the camera
4. View classification results with confidence scores

## 📂 Expected Classes

- background
- boxcutter
- envelope
- mouse
- sanitizer
- smartphone
- stapler

## 🛠️ Tech Stack

- **Frontend**: Vue 2 (CDN), Axios, Modern CSS
- **Backend**: Flask, YOLO (Ultralytics), OpenCV
- **Model**: YOLOv8 Classification

## 🐛 Troubleshooting

### PyTorch 2.6 Compatibility Issues

If you get "Unexpected type numpy.ndarray" errors:

**Option 1: Use the fix script (Windows)**
```bash
fix_torch.bat
```

**Option 2: Manual fix**
```bash
pip uninstall torch torchvision -y
pip install torch==2.0.1 torchvision==0.15.2
pip install -r requirements.txt
```

### Other Issues

- **Model not found**: Make sure your trained model is at `../runs/classify/train2/weights/best.pt`
- **Camera not working**: Check if your camera is connected and not used by other apps
- **Port 5001 in use**: Change the port in `app.py` if needed

## 📝 Notes

- The server runs on port 5001
- Camera classification opens in a separate OpenCV window
- All files are self-contained (no build process needed)
- Uses CDN for Vue 2 and Axios (no npm required)
- Compatible with PyTorch 2.0.1 and torchvision 0.15.2