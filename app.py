from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
from PIL import Image
import io
import base64
import threading
import os
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)

# Simple globals
model = None
camera_thread = None
stop_camera = threading.Event()

def get_model():
    """Load your trained YOLO model"""
    global model
    if model is None:
        model_path = "runs/detect/train/weights/best.pt"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Trained model not found at {model_path}. Please train your model first.")
        
        # Use your trained model for office items
        model = YOLO(model_path)
        print(f" Loaded trained model with {len(model.names)} classes:")
        for i, name in model.names.items():
            print(f"   {i}: {name}")
    return model

def image_to_base64(img):
    """Convert image to base64 for web display"""
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"

def camera_loop():
    """Simple camera detection loop"""
    yolo = get_model()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Camera not available")
        return
    
    cv2.namedWindow("Live Detection - Press 'q' to stop", cv2.WINDOW_NORMAL)
    
    while not stop_camera.is_set():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Run detection
        results = yolo.predict(frame, conf=0.35, verbose=False)
        if results:
            frame = results[0].plot()
            
        cv2.imshow("Live Detection - Press 'q' to stop", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

@app.route('/')
def home():
    return send_from_directory('ui', 'index.html')

@app.route('/<path:filename>')
def files(filename):
    return send_from_directory('ui', filename)

@app.route('/api/detect', methods=['POST'])
def detect():
    try:
        # Get uploaded image
        if 'image' not in request.files:
            return jsonify({'error': 'No image'}), 400
        
        image = Image.open(request.files['image'].stream)
        
        # Run detection
        yolo = get_model()
        img_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        results = yolo.predict(img_array, conf=0.25, verbose=False)
        
        if not results:
            return jsonify({'image': image_to_base64(image), 'objects': []})
        
        # Get annotated image
        result = results[0]
        annotated = result.plot()
        annotated_img = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        
        # Extract objects
        objects = []
        if result.boxes is not None:
            for i, (cls, conf) in enumerate(zip(result.boxes.cls, result.boxes.conf)):
                objects.append({
                    'id': i,
                    'name': yolo.names[int(cls)],
                    'confidence': round(float(conf) * 100, 1)
                })
        
        return jsonify({
            'image': image_to_base64(annotated_img),
            'objects': objects
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/start-camera', methods=['POST'])
def start_camera():
    global camera_thread
    
    if camera_thread and camera_thread.is_alive():
        return jsonify({'status': 'Camera already running'})
    
    stop_camera.clear()
    camera_thread = threading.Thread(target=camera_loop, daemon=True)
    camera_thread.start()
    
    return jsonify({'status': 'Camera started'})

@app.route('/api/stop-camera', methods=['POST'])
def stop_camera_route():
    stop_camera.set()
    if camera_thread:
        camera_thread.join(timeout=2)
    return jsonify({'status': 'Camera stopped'})


if __name__ == '__main__':
    print("🎯 Office Item Detector Web Server")
    print("=" * 40)
    print("Using trained model: runs/detect/train/weights/best.pt")
    print("Server starting at: http://localhost:5000")
    print("=" * 40)
    app.run(debug=True, host='0.0.0.0', port=5000)