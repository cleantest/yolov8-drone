import threading
import time
import os
from pathlib import Path
from typing import Tuple, List

import cv2
import numpy as np
from PIL import Image

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

try:
    import gradio as gr
except Exception:
    gr = None

# Path to the local model file (project root)
MODEL_PATH = Path(__file__).parent / "yolov8n.pt"

# Global model singleton and live thread control
_MODEL = None
_live_thread = None
_live_stop_event = threading.Event()


def load_model():
    global _MODEL
    if _MODEL is None:
        if YOLO is None:
            raise RuntimeError("ultralytics package is not installed. Install via: pip install ultralytics")
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Place your yolov8 model file there.")
        _MODEL = YOLO(str(MODEL_PATH))
    return _MODEL


def predict_image(img: Image.Image, conf: float = 0.25) -> Tuple[Image.Image, str]:
    """Run detection on a PIL image and return annotated image and a small text summary."""
    model = load_model()
    # Convert PIL to numpy BGR for ultralytics if needed
    img_np = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # Run prediction (synchronous)
    results = model.predict(source=img_np, conf=conf, verbose=False)
    if len(results) == 0:
        return img, "No results"

    r = results[0]
    # ultralytics' .plot() returns annotated image in RGB
    annotated = r.plot()
    # Convert to PIL Image
    annotated_pil = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))

    # Build text summary of detections
    labels = []
    if hasattr(r, 'boxes') and r.boxes is not None and len(r.boxes) > 0:
        for box, cls in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist()):
            name = model.names[int(cls)] if model.names and int(cls) in model.names else str(int(cls))
            labels.append(name)
    summary = "Detected: " + (", ".join(labels) if labels else "(none)")
    return annotated_pil, summary


def _live_loop(conf: float = 0.35, camera_index: int = 0):
    """Background loop that opens a webcam window and runs detection until stopped."""
    try:
        model = load_model()
    except Exception as e:
        print("Live detection failed to load model:", e)
        return

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Could not open camera")
        return

    window_name = "Live Object Detection - press 'q' to stop"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while not _live_stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, conf=conf, verbose=False)
        if len(results) > 0:
            annotated = results[0].plot()
        else:
            annotated = frame

        # Show frame
        cv2.imshow(window_name, annotated)

        # Allow quitting with 'q' in the cv2 window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    _live_stop_event.clear()
    cap.release()
    cv2.destroyWindow(window_name)


def start_live(conf: float = 0.35, camera_index: int = 0) -> str:
    """Start background live detection using OpenCV window."""
    global _live_thread, _live_stop_event
    if _live_thread is not None and _live_thread.is_alive():
        return "Live detection already running"

    _live_stop_event.clear()
    _live_thread = threading.Thread(target=_live_loop, args=(conf, camera_index), daemon=True)
    _live_thread.start()
    # Give thread a moment to start and check camera
    time.sleep(0.5)
    return "Live detection started. Focus the OpenCV window and press 'q' to stop, or use the Stop button."


def stop_live() -> str:
    global _live_thread, _live_stop_event
    if _live_thread is None or not _live_thread.is_alive():
        return "Live detection is not running"
    _live_stop_event.set()
    # Wait briefly for thread to finish
    _live_thread.join(timeout=3.0)
    if _live_thread.is_alive():
        return "Stop requested; thread did not exit promptly. You may need to close the OpenCV window manually."
    return "Live detection stopped"


def build_ui():
    if gr is None:
        raise RuntimeError("gradio is not installed. Install via: pip install gradio")

    with gr.Blocks(title="Office Item Detector") as demo:
        gr.Markdown("# Office Item Detector\nUpload an image to detect office items, or run a live camera feed.")

        with gr.Tabs():
            with gr.TabItem("Image Upload"):
                img_in = gr.Image(type="pil", label="Upload image")
                conf_slider = gr.Slider(minimum=0.05, maximum=0.9, step=0.01, value=0.25, label="Confidence")
                predict_btn = gr.Button("Detect")
                img_out = gr.Image(type="pil", label="Annotated")
                txt_out = gr.Textbox(label="Summary")

                def _run_image(img, conf):
                    if img is None:
                        return None, "No image provided"
                    try:
                        annotated, summary = predict_image(img, conf)
                        return annotated, summary
                    except Exception as e:
                        return None, f"Error: {e}"

                predict_btn.click(_run_image, inputs=[img_in, conf_slider], outputs=[img_out, txt_out])

            with gr.TabItem("Live Camera"):
                conf_live = gr.Slider(minimum=0.05, maximum=0.9, step=0.01, value=0.35, label="Confidence")
                cam_index = gr.Number(value=0, label="Camera index (usually 0)")
                start_btn = gr.Button("Start Live")
                stop_btn = gr.Button("Stop Live")
                live_status = gr.Textbox(label="Live status", value="Stopped")

                def _start(conf, cam):
                    try:
                        msg = start_live(conf, int(cam))
                        return msg
                    except Exception as e:
                        return f"Error starting live: {e}"

                def _stop():
                    return stop_live()

                start_btn.click(_start, inputs=[conf_live, cam_index], outputs=[live_status])
                stop_btn.click(_stop, outputs=[live_status])

        gr.Markdown("---\nModel file used: %s" % MODEL_PATH.name)

    return demo


if __name__ == "__main__":
    # When executed directly, launch the Gradio UI
    try:
        app = build_ui()
        app.launch(server_name="0.0.0.0", share=True)
    except Exception as exc:
        print("Failed to start UI:", exc)
        print("Make sure required packages are installed. A minimal requirements file is provided.")
