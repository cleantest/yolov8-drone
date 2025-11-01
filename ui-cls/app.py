"""
YOLO Classification Web Server

This Flask app provides a web interface for image classification using YOLO.
It supports both image upload and live camera classification.

Author: Created for office item classification project
Date: 2024
"""

from flask import Flask, request, jsonify, send_from_directory
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

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

# Set environment variable to allow unsafe loading (PyTorch 2.6 compatibility)
os.environ['TORCH_WEIGHTS_ONLY'] = 'False'

# Fix for numpy array issue in torchvision transforms
def patch_torchvision():
    """Patch torchvision to handle numpy arrays properly"""
    try:
        import torchvision.transforms._functional_pil as F_pil
        from PIL import Image
        
        # Store original function
        original_get_dimensions = F_pil.get_dimensions
        
        def patched_get_dimensions(img):
            # Convert numpy array to PIL Image if needed
            if isinstance(img, np.ndarray):
                if img.ndim == 3 and img.shape[2] == 3:  # RGB image
                    img = Image.fromarray(img.astype('uint8'))
                elif img.ndim == 2:  # Grayscale image
                    img = Image.fromarray(img.astype('uint8'), mode='L')
                else:
                    # Fallback: try to convert anyway
                    img = Image.fromarray(img.astype('uint8'))
            
            return original_get_dimensions(img)
        
        # Apply the patch
        F_pil.get_dimensions = patched_get_dimensions
        print("✅ Applied torchvision numpy array compatibility patch")
        
    except Exception as e:
        print(f"⚠️  Could not apply torchvision patch: {e}")

# Apply the patch at import time
patch_torchvision()

# Initialize Flask app with CORS for web requests
app = Flask(__name__)
CORS(app)

# Global variables - keeping it simple for this demo
model = None  # Will hold our loaded YOLO model
camera_thread = None  # Thread for camera processing
stop_camera = threading.Event()  # Signal to stop camera thread

def get_model():
    """
    Load the YOLO classification model.
    
    This function tries to find a trained model in common locations,
    falling back to a pretrained model if needed. It also handles
    PyTorch 2.6 compatibility issues with model loading.
    
    Returns:
        YOLO: Loaded YOLO classification model
    """
    global model
    
    # Only load once - reuse the same model instance
    if model is None:
        # Find the trained model in possible locations
        possible_paths = [
            os.path.join("..", "runs", "classify", "train2", "weights", "best.pt"),
            os.path.join("runs", "classify", "train2", "weights", "best.pt"),
            os.path.join("..", "..", "runs", "classify", "train2", "weights", "best.pt"),
            "runs/classify/train2/weights/best.pt",
            "../runs/classify/train2/weights/best.pt"
        ]
        
        model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if model_path is None or not os.path.exists(model_path):
            raise FileNotFoundError(f"Classification model not found at {model_path}. Please train your model first.")
        
        # Handle PyTorch 2.6 compatibility - use weights_only=False
        try:
            print("   Loading YOLO model with PyTorch 2.6 compatibility...")
            
            # Store original torch.load function
            original_load = torch.load
            
            # Create a wrapper that always uses weights_only=False
            def safe_load(*args, **kwargs):
                kwargs['weights_only'] = False
                return original_load(*args, **kwargs)
            
            # Temporarily replace torch.load
            torch.load = safe_load
            
            # Load the YOLO model
            model = YOLO(model_path)
            
            # Restore original torch.load
            torch.load = original_load
            
            print("   ✅ Model loaded successfully")
            
        except Exception as e:
            # Make sure to restore torch.load even if there's an error
            torch.load = original_load
            print(f"   ❌ Model loading failed: {e}")
            raise Exception(f"Could not load YOLO model: {e}")
        
        print(f"✅ Successfully loaded model: {model_path}")
        print(f"📋 Available classes: {list(model.names.values())}")
    
    return model

def image_to_base64(img):
    """
    Convert a PIL Image to base64 string for web display.
    
    This is needed because we can't directly send image files to the browser,
    so we encode them as base64 strings that can be embedded in HTML.
    
    Args:
        img (PIL.Image): The image to convert
        
    Returns:
        str: Base64 encoded image with data URL prefix
    """
    # Create a bytes buffer to hold the image data
    buffer = io.BytesIO()
    
    # Save the image to the buffer as PNG
    img.save(buffer, format="PNG")
    
    # Encode as base64 and create a data URL
    img_data = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_data}"

def camera_loop():
    """
    Main loop for live camera classification.
    Captures frames and runs YOLO classification continuously.
    """
    yolo = get_model()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print(" Could not access camera")
        return
    
    print("📹 Camera started successfully - Press 'q' in the camera window to stop")
    
    window_name = "Live Classification - Press 'q' to stop"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    while not stop_camera.is_set():
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from camera")
            break
        
        try:
            # Convert BGR to RGB and to PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            # Manual inference path: preprocess and run model directly
            try:
                from torchvision import transforms

                preprocess = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

                input_tensor = preprocess(pil_image).unsqueeze(0)

                # Move tensor to model device
                try:
                    model_params = list(yolo.model.parameters())
                    device = model_params[0].device if len(model_params) > 0 else torch.device('cpu')
                except Exception:
                    device = torch.device('cpu')

                input_tensor = input_tensor.to(device)

                with torch.no_grad():
                    outputs = yolo.model(input_tensor)

                # Extract logits/tensor
                logits = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
                probs_tensor = torch.softmax(logits, dim=1)[0]

                top5conf, top5 = torch.topk(probs_tensor, min(5, probs_tensor.numel()))
                top1 = top5[0]
                top1conf = top5conf[0]

                # Build a lightweight result object compatible with existing code
                class Probs:
                    pass

                class Result:
                    pass

                p = Probs()
                p.top5 = top5
                p.top5conf = top5conf
                p.top1 = top1
                p.top1conf = top1conf

                res = Result()
                res.probs = p
                results = [res]

            except Exception as inner_e:
                print(f"Manual inference failed: {inner_e}")
                # Fallback: attempt predict with temporary file as before
                try:
                    temp_path = "temp_frame.jpg"
                    pil_image.save(temp_path)
                    results = yolo.predict(source=temp_path, verbose=False)
                finally:
                    try:
                        os.remove(temp_path)
                    except:
                        pass

            # Check if we got valid results and draw
            if results and len(results) > 0 and hasattr(results[0], 'probs'):
                top_class_idx = results[0].probs.top1
                # top1 may be a tensor -> convert
                try:
                    top_idx = int(top_class_idx.item()) if hasattr(top_class_idx, 'item') else int(top_class_idx)
                except Exception:
                    top_idx = int(top_class_idx)

                try:
                    confidence = float(results[0].probs.top1conf.item()) if hasattr(results[0].probs.top1conf, 'item') else float(results[0].probs.top1conf)
                except Exception:
                    confidence = 0.0

                class_name = yolo.names[top_idx]
                text = f"{class_name}: {confidence:.2f}"
                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow(window_name, frame)

        except Exception as e:
            print(f"Error in camera loop: {e}")
        
        # Check for 'q' key to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up camera resources
    cap.release()
    cv2.destroyAllWindows()
    print("📹 Camera stopped and resources cleaned up")

# Web routes for serving the frontend
@app.route('/')
def home():
    """Serve the main HTML page"""
    return send_from_directory('.', 'index.html')

@app.route('/<path:filename>')
def files(filename):
    """Serve static files (CSS, JS, etc.)"""
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
        
        # Extract the top 5 predictions
        predictions = []
        num_predictions = min(5, len(probs.top5))
        
        for i in range(num_predictions):
            try:
                # Handle both tensor and numpy array types
                class_idx_raw = probs.top5[i]
                confidence_raw = probs.top5conf[i]
                
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
                print(f"Error processing prediction {i}: {pred_error}")
                print(f"  class_idx type: {type(probs.top5[i])}")
                print(f"  confidence type: {type(probs.top5conf[i])}")
                continue
        
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

@app.route('/api/start-camera', methods=['POST'])
def start_camera():
    """
    Start the live camera classification in a separate thread.
    
    This opens a camera window that shows live classification results.
    Only one camera session can run at a time.
    """
    global camera_thread
    
    # Check if camera is already running
    if camera_thread and camera_thread.is_alive():
        return jsonify({'status': 'Camera already running'})
    
    # Reset the stop signal and start a new camera thread
    stop_camera.clear()
    camera_thread = threading.Thread(target=camera_loop, daemon=True)
    camera_thread.start()
    
    return jsonify({'status': 'Camera started'})

@app.route('/api/stop-camera', methods=['POST'])
def stop_camera_route():
    """
    Stop the live camera classification.
    
    This signals the camera thread to stop and waits for it to finish.
    """
    # Signal the camera thread to stop
    stop_camera.set()
    
    # Wait for the thread to finish (with timeout to avoid hanging)
    if camera_thread:
        camera_thread.join(timeout=2)
    
    return jsonify({'status': 'Camera stopped'})

@app.route('/api/classes', methods=['GET'])
def get_classes():
    """
    Get the list of classes that the model can classify.
    
    This is useful for the frontend to display available classes.
    """
    try:
        yolo = get_model()
        class_names = list(yolo.names.values())
        return jsonify({'classes': class_names})
    except Exception as e:
        print(f"Error getting classes: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Main entry point
if __name__ == '__main__':
    print("🎯 YOLO Classification Web Server")
    print("=" * 50)
    print("📂 Expected classes: background, boxcutter, envelope, mouse,")
    print("                    sanitizer, smartphone, stapler")
    print("🌐 Server will be available at: http://localhost:5001")
    print("📹 Camera classification opens in a separate OpenCV window")
    print("⚠️  Make sure your camera is connected for live classification")
    print("=" * 50)
    
    # Test model loading at startup
    try:
        print("\n🔍 Testing model loading...")
        test_model = get_model()
        print(f"✅ Model loaded successfully with {len(test_model.names)} classes")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        print("Please check your model file and try again.")
        exit(1)
    
    print("\n🚀 Starting Flask server...")
    
    # Start the Flask development server
    # debug=True enables auto-reload during development
    # host='0.0.0.0' allows access from other devices on the network
    app.run(debug=True, host='0.0.0.0', port=5001)