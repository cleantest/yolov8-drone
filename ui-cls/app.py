"""
YOLO Classification Web Server

Flask backend for image classification using YOLO models.
Supports both image upload and live camera streaming with real-time classification.
"""

from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS
import cv2
import numpy as np
from PIL import Image
import io
import base64
import threading
import os
import torch
from ultralytics import YOLO
import warnings
import os

warnings.filterwarnings("ignore", category=UserWarning)
os.environ['TORCH_WEIGHTS_ONLY'] = 'False'

def patch_torchvision():
    """Fix torchvision transforms to handle numpy arrays"""
    try:
        import torchvision.transforms._functional_pil as F_pil
        from PIL import Image
        
        original_get_dimensions = F_pil.get_dimensions
        
        def patched_get_dimensions(img):
            if isinstance(img, np.ndarray):
                if img.ndim == 3 and img.shape[2] == 3:
                    img = Image.fromarray(img.astype('uint8'))
                elif img.ndim == 2:
                    img = Image.fromarray(img.astype('uint8'), mode='L')
                else:
                    img = Image.fromarray(img.astype('uint8'))
            return original_get_dimensions(img)
        
        F_pil.get_dimensions = patched_get_dimensions

        
    except Exception as e:
        pass

patch_torchvision()

app = Flask(__name__)
CORS(app)

model = None
camera_thread = None
stop_camera = threading.Event()

camera_running = False
cap = None
latest_detection = {"class": "", "confidence": 0, "timestamp": 0}
detection_history = []

def get_model():
    """Load the YOLO classification model"""
    global model
    
    if model is None:
        possible_paths = [
            "runs/classify/train2/weights/best.pt",
            "../runs/classify/train2/weights/best.pt"
        ]
        
        model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if model_path is None:
            raise FileNotFoundError("Classification model not found. Please train your model first.")
        
        try:
            original_load = torch.load
            
            def safe_load(*args, **kwargs):
                kwargs['weights_only'] = False
                return original_load(*args, **kwargs)
            
            torch.load = safe_load
            model = YOLO(model_path)
            torch.load = original_load
            
        except Exception as e:
            torch.load = original_load
            raise Exception(f"Could not load YOLO model: {e}")
    
    return model

def image_to_base64(img):
    """Convert PIL Image to base64 string for web display"""
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    img_data = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_data}"

def classify_frame(frame):
    try:
        yolo = get_model()
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb.astype('uint8'))
        
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')

        temp_path = "temp_camera_frame.jpg"
        pil_image.save(temp_path)
        
        try:
            from ultralytics.models.yolo.classify.predict import ClassificationPredictor
            
            original_preprocess = ClassificationPredictor.preprocess
            
            def patched_preprocess(self, img):
                if isinstance(img, list):
                    processed_imgs = []
                    for im in img:
                        if isinstance(im, np.ndarray):
                            if im.ndim == 3:
                                im = Image.fromarray(im.astype('uint8'))
                            elif im.ndim == 2:
                                im = Image.fromarray(im.astype('uint8'), mode='L')
                        processed_imgs.append(im)
                    img = processed_imgs
                elif isinstance(img, np.ndarray):
                    if img.ndim == 3:
                        img = Image.fromarray(img.astype('uint8'))
                    elif img.ndim == 2:
                        img = Image.fromarray(img.astype('uint8'), mode='L')
                
                return original_preprocess(self, img)
            
            ClassificationPredictor.preprocess = patched_preprocess
            results = yolo.predict(temp_path, verbose=False)
            ClassificationPredictor.preprocess = original_preprocess
            
            if results and hasattr(results[0], "probs"):
                probs = results[0].probs
                idx = int(probs.top1.item()) if hasattr(probs.top1, "item") else int(probs.top1)
                conf = float(probs.top1conf.item()) if hasattr(probs.top1conf, "item") else float(probs.top1conf)
                class_name = yolo.names[idx]
                return class_name, conf
                
        finally:
            try:
                os.remove(temp_path)
            except:
                pass

    except Exception as e:
        print(f"Error in camera classification: {e}")
    
    return "unknown", 0.0

def generate_frames():
    """Generate camera frames for browser streaming with detection"""
    global cap, camera_running, latest_detection, detection_history
    
    yolo = get_model()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Could not access camera")
        camera_running = False
        return
    

    frame_count = 0
    last_high_confidence_time = 0
    
    while camera_running:
        success, frame = cap.read()
        if not success:
            print("Failed to read camera frame")
            break
        
        frame_count += 1
        
        try:
            if frame_count % 10 == 0:
                class_name, confidence = classify_frame(frame)
                
                if confidence > 0.8 and class_name.lower() != "background":
                    import time
                    current_timestamp = time.time()
                    
                    if current_timestamp - last_high_confidence_time > 3:
                        latest_detection = {
                            "class": class_name,
                            "confidence": round(confidence * 100, 1),
                            "timestamp": current_timestamp
                        }
                        
                        detection_history.append(latest_detection.copy())
                        
                        if len(detection_history) > 10:
                            detection_history.pop(0)
                        
                        last_high_confidence_time = current_timestamp

                        camera_running = False
                        break
                
                if confidence > 0.1:
                    label = f"{class_name}: {confidence*100:.1f}%"
                    
                    if class_name.lower() == "background":
                        color = (128, 128, 128)
                    elif confidence > 0.8:
                        color = (0, 255, 0)
                    else:
                        color = (0, 255, 255)
                    
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                    cv2.rectangle(frame, (10, 10), (text_size[0] + 20, text_size[1] + 30), (0, 0, 0), -1)
                    cv2.putText(frame, label, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    
                    if confidence > 0.8 and class_name.lower() != "background":
                        cv2.putText(frame, "HIGH CONFIDENCE!", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    elif class_name.lower() == "background":
                        cv2.putText(frame, "Searching for objects...", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
        
        except Exception as e:
            print(f"Classification error: {e}")
            cv2.putText(frame, "Classification Error", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            continue

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               buffer.tobytes() + b"\r\n")

    if cap is not None:
        cap.release()
        cap = None




@app.route('/')
def home():
    return send_from_directory('.', 'index.html')

@app.route('/<path:filename>')
def files(filename):
    return send_from_directory('.', filename)

# API endpoints
@app.route('/api/classify', methods=['POST'])
def classify():
    """
    Classify an uploaded image and return predictions.
    
    Expects a multipart form with an 'image' file.
    Returns JSON with the image (as base64) and top 5 predictions.
    """
    try:
        # Check if an image was uploaded
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        # Load the uploaded image
        uploaded_file = request.files['image']
        
        # Save the uploaded file temporarily
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            uploaded_file.save(temp_file.name)
            temp_path = temp_file.name
        
        try:
            # Load image for display
            image = Image.open(temp_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Get our model and run classification
            yolo = get_model()
            
            # Use a robust prediction method that bypasses the numpy array issue
            try:
                # Patch the YOLO predictor to handle numpy arrays properly
                from ultralytics.models.yolo.classify.predict import ClassificationPredictor
                
                # Store original preprocess method
                original_preprocess = ClassificationPredictor.preprocess
                
                def patched_preprocess(self, img):
                    """Patched preprocess that ensures PIL Images"""
                    # Convert numpy arrays to PIL Images before processing
                    if isinstance(img, list):
                        processed_imgs = []
                        for im in img:
                            if isinstance(im, np.ndarray):
                                # Convert numpy array to PIL Image
                                if im.ndim == 3:
                                    im = Image.fromarray(im.astype('uint8'))
                                elif im.ndim == 2:
                                    im = Image.fromarray(im.astype('uint8'), mode='L')
                            processed_imgs.append(im)
                        img = processed_imgs
                    elif isinstance(img, np.ndarray):
                        if img.ndim == 3:
                            img = Image.fromarray(img.astype('uint8'))
                        elif img.ndim == 2:
                            img = Image.fromarray(img.astype('uint8'), mode='L')
                    
                    # Call original preprocess with PIL Images
                    return original_preprocess(self, img)
                
                # Apply the patch
                ClassificationPredictor.preprocess = patched_preprocess
                
                # Now try prediction
                results = yolo.predict(temp_path, verbose=False)
                
                # Restore original method
                ClassificationPredictor.preprocess = original_preprocess
                
            except Exception as e:
                print(f"Patched prediction failed: {e}")
                
                # Fallback: Use manual classification
                try:
                    print("Trying manual classification...")
                    
                    # Load and preprocess image manually
                    pil_img = Image.open(temp_path).convert('RGB')
                    pil_img = pil_img.resize((224, 224))
                    
                    # Convert to tensor manually
                    import torchvision.transforms as transforms
                    
                    transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
                    
                    tensor_img = transform(pil_img).unsqueeze(0)
                    
                    # Run inference directly on the model
                    with torch.no_grad():
                        outputs = yolo.model(tensor_img)
                        
                    # Create a mock result object
                    class MockResult:
                        def __init__(self, outputs, names):
                            self.probs = MockProbs(outputs, names)
                    
                    class MockProbs:
                        def __init__(self, outputs, names):
                            if isinstance(outputs, tuple):
                                outputs = outputs[0]
                            probs = torch.softmax(outputs, dim=1)[0]
                            self.top5conf, self.top5 = torch.topk(probs, min(5, len(probs)))
                            self.top1 = self.top5[0]
                            self.top1conf = self.top5conf[0]
                    
                    results = [MockResult(outputs, yolo.names)]
                    print("✅ Manual classification successful")
                    
                except Exception as e2:
                    print(f"Manual classification failed: {e2}")
                    raise Exception(f"All classification methods failed: {e}, {e2}")
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_path)
            except:
                pass
        
        # Handle case where no predictions were made
        if not results:
            return jsonify({
                'image': image_to_base64(image),
                'predictions': []
            })
        
        result = results[0]
        
        # Check if we have classification probabilities
        if not hasattr(result, 'probs') or result.probs is None:
            return jsonify({
                'image': image_to_base64(image),
                'predictions': []
            })
        
        probs = result.probs
        
        # Check if we have valid predictions
        if not hasattr(probs, 'top5') or len(probs.top5) == 0:
            return jsonify({
                'image': image_to_base64(image),
                'predictions': []
            })
        
        # Extract only the top prediction (highest confidence)
        predictions = []
        
        try:
            # Get the top prediction only
            class_idx_raw = probs.top1
            confidence_raw = probs.top1conf
            
            # Convert to Python native types
            if hasattr(class_idx_raw, 'item'):
                class_idx = int(class_idx_raw.item())
            else:
                class_idx = int(class_idx_raw)
            
            if hasattr(confidence_raw, 'item'):
                confidence = float(confidence_raw.item())
            else:
                confidence = float(confidence_raw)
            
            # Get class name safely
            class_name = str(yolo.names[class_idx])
            
            predictions.append({
                'class': class_name,
                'confidence': round(confidence * 100, 1)  # Convert to percentage
            })
            
        except Exception as pred_error:
            print(f"Error processing top prediction: {pred_error}")
            print(f"  class_idx type: {type(probs.top1)}")
            print(f"  confidence type: {type(probs.top1conf)}")
        
        # Return the results
        return jsonify({
            'image': image_to_base64(image),
            'predictions': predictions
        })
        
    except Exception as e:
        # Log detailed error information
        print(f"Classification error: {str(e)}")
        print(f"Error type: {type(e)}")
        
        # Import traceback for detailed error info
        import traceback
        print("Full traceback:")
        traceback.print_exc()
        
        return jsonify({'error': f"Classification failed: {str(e)}"}), 500

@app.route('/video_feed')
def video_feed():
    global camera_running
    if camera_running:
        return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, "Click Start Camera", (150, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(placeholder, "to begin live classification", (120, 260), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        ret, buffer = cv2.imencode('.jpg', placeholder)
        return Response(buffer.tobytes(), mimetype='image/jpeg')

@app.route('/api/start-camera', methods=['POST'])
def start_camera():
    global camera_running
    camera_running = True

    return jsonify({'status': 'Camera started'})

@app.route('/api/stop-camera', methods=['POST'])
def stop_camera_api():
    global camera_running, cap
    camera_running = False
    if cap is not None:
        cap.release()
        cap = None

    return jsonify({'status': 'Camera stopped'})

@app.route('/api/camera-status', methods=['GET'])
def get_camera_status():
    global camera_running
    return jsonify({'running': camera_running})

@app.route('/api/detections', methods=['GET'])
def get_detections():
    global detection_history, latest_detection
    return jsonify({
        'latest': latest_detection,
        'history': detection_history
    })

@app.route('/api/clear-detections', methods=['POST'])
def clear_detections():
    global detection_history, latest_detection
    detection_history.clear()
    latest_detection = {"class": "", "confidence": 0, "timestamp": 0}
    return jsonify({'status': 'Detections cleared'})

@app.route('/api/classes', methods=['GET'])
def get_classes():
    try:
        yolo = get_model()
        class_names = list(yolo.names.values())
        return jsonify({'classes': class_names})
    except Exception as e:
        print(f"Error getting classes: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    get_model()
    app.run(debug=True, host='0.0.0.0', port=5001)