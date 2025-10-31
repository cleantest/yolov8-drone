#!/usr/bin/env python3
"""
Simple YOLO Classification Server Launcher

Quick start script for the Vue 2 + Flask classification app
Usage: python run.py

Created for office item classification project - 2024
"""

import os
import sys

def check_model():
    """Check if the required model exists"""
    # Try multiple possible paths
    possible_paths = [
        os.path.join("..", "runs", "classify", "train2", "weights", "best.pt"),
        os.path.join("runs", "classify", "train2", "weights", "best.pt"),
        os.path.join("..", "..", "runs", "classify", "train2", "weights", "best.pt"),
        "runs/classify/train2/weights/best.pt",
        "../runs/classify/train2/weights/best.pt"
    ]
    
    print(f"🔍 Current directory: {os.getcwd()}")
    print("🔍 Searching for model in these locations:")
    
    for path in possible_paths:
        abs_path = os.path.abspath(path)
        exists = os.path.exists(path)
        print(f"   {path} -> {abs_path} ({'✅' if exists else '❌'})")
        
        if exists:
            print(f"✅ Found model: {path}")
            return True, path
    
    print("\n❌ Model not found in any location!")
    print("📝 Please ensure your model is trained and available")
    print("   Expected location: runs/classify/train2/weights/best.pt")
    return False, None

def main():
    """Start the classification server"""
    print("🎯 YOLO Classification Server (Vue 2 + Flask)")
    print("=" * 50)
    
    # Check for required model
    model_found, model_path = check_model()
    if not model_found:
        return
    
    print("\n🚀 Starting Flask backend server...")
    print("🌐 Open your browser to: http://localhost:5001")
    print("📹 Camera opens in separate OpenCV window")
    print("⏹️  Press Ctrl+C to stop")
    print("=" * 50)
    
    try:
        # Start the Flask app
        from app import app
        app.run(debug=True, host='0.0.0.0', port=5001)
        
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("Install requirements: pip install -r requirements.txt")
    except Exception as e:
        print(f"\n❌ Server error: {e}")

if __name__ == '__main__':
    main()