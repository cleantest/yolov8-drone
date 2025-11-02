# YOLOv8 Image Classification Project

This repository contains a YOLOv8-based image classification system with a web interface for both real-time camera classification and image upload classification.

## Getting Started

### Download and Setup

1. Clone the repository:
```bash
git clone https://github.com/cleantest/yolov8-drone.git
cd yolov8-drone
```

2. Download the classification dataset:
   - Go to [https://github.com/cleantest/yolov8-drone/releases/tag/cls-dataset.v10]
   - Download the `cls_dataset.zip` file
   - Extract the contents in the root of the project:
```bash
# After downloading, your zip file should be in the downloads folder


# Linux/Mac
unzip ~/Downloads/cls_dataset.zip
```

The dataset contains pre-organized training, validation, and test images for classification.

## Project Structure

```
.
├── cls_dataset/          # Classification dataset directory
│   ├── train/           # Training images
│   ├── valid/          # Validation images
│   └── test/           # Test images
└── ui-cls/             # Web interface application
    ├── app.py          # Flask web server
    ├── templates/      # HTML templates
    └── static/        # Static assets (CSS, JS)
```

## Prerequisites

- Python 3.8 or later
- CUDA-compatible GPU (recommended for faster inference)
- Webcam (for real-time classification)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/cleantest/yolov8-classification.git
cd yolov8-classification
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

Required packages include:
- flask==2.3.3
- flask-cors==4.0.0
- ultralytics==8.0.20
- opencv-python==4.8.0.76
- pillow==9.5.0
- numpy==1.24.3
- torch==2.0.1
- torchvision==0.15.2
- torchaudio==2.0.2


### Running the Web Interface

1. Navigate to the ui-cls directory:
```bash
cd ui-cls
```

2. Start the Flask application:
```bash
python app.py
```

3. Open your web browser and visit:
```
http://localhost:5001
```

### Features

1. **Image Upload Classification**
   - Click the "Choose File" button
   - Select an image
   - Click "Upload" to see the classification results

2. **Real-time Camera Classification**
   - Click "Start Camera" to begin real-time classification
   - The classification results will be displayed in real-time
   - Click "Stop Camera" to end the session

## Expected Output

The classification system will output:
- Class name (e.g., "keyboard", "mouse", etc.)
- Confidence score (0-100%)
- Real-time classification results for camera feed
- Classification results for uploaded images with their confidence scores

## Troubleshooting

### Common Issues and Solutions

1. **Camera Not Working**
   - Ensure your webcam is properly connected
   - Check if another application is using the camera
   - Try restarting the Flask application
   - Verify camera permissions in your browser

2. **Model Loading Errors**
   - Confirm the model path is correct in `app.py`
   - Ensure the model file exists in the specified location
   - Check if CUDA is available for GPU inference

3. **Low Classification Accuracy**
   - Verify input image size matches training size (224x224)
   - Check if the image preprocessing matches training preprocessing
   - Ensure the model is loaded with the correct weights
   - Consider retraining with more diverse data

4. **Web Interface Issues**
   - Clear browser cache and refresh
   - Check Flask server logs for errors
   - Verify all static files are properly loaded
   - Ensure correct port (5001) is available

### Debug Mode

To run the application in debug mode with detailed logging:
```bash
python app.py --debug
```

## Additional Resources

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [OpenCV Python Tutorial](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)

## Contributing

Feel free to submit issues and enhancement requests!