#  Office Item Detector

AI-powered object detection system built with YOLOv8 for identifying office items through a web interface.

![Office Item Detector](https://img.shields.io/badge/AI-YOLOv8-blue) ![Python](https://img.shields.io/badge/Python-3.8+-green) ![License](https://img.shields.io/badge/License-MIT-yellow)

##  Detectable Objects

- Envelope, Mouse, Stapler, Sanitizer, Boxcutter, Smartphone

##  Quick Start

### Prerequisites
- Python 3.8+
- Webcam (for live detection)

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd office-item-detector
   ```

2. **Download the dataset**
   - Go to **Releases** section on the main branch
   - Download the dataset zip file (dataset-v1)
   - Extract and add the train, test, valid folders to the project root

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the model** (optional - if you want to retrain)
   ```bash
   python train.py
   ```

5. **Start the web interface**
   ```bash
   python app.py
   ```

6. **Open your browser**
   - Go to: http://localhost:5000
   - Upload images or use live camera detection

##  Project Structure

```
office-item-detector/
├──  ui/                   # Web interface files
├──  train/                # Training dataset
├──  valid/                # Validation dataset  
├──  test/                 # Test dataset
├──  runs/                 # Training/detection results
├── app.py                   # Flask web server
├── train.py                 # Model training script
├── data.yaml                # Dataset configuration
└── requirements.txt         # Dependencies
```

##  Usage

### Web Interface
1. **Upload Images**: Drag & drop or click to select images
2. **Live Camera**: Real-time detection using webcam
3. **View Results**: See detected objects with confidence scores

### Training New Model
1. Update `data.yaml` with your classes
2. Place images in train/valid/test folders
3. Run `python train.py`
4. Restart web server to use new model

##  Troubleshooting

- **Server won't start**: Install requirements with `pip install -r requirements.txt`
- **Camera not working**: Check camera and browser permissions and close other camera apps
- **Model not found**: Make sure training completed successfully



