from ultralytics import YOLO
import cv2

# Load your trained model
model = YOLO('runs/detect/train/weights/best.pt')

# Open webcam
cap = cv2.VideoCapture(0)  # 0 = default webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO inference
    results = model(frame, stream=True)

    # Visualize results
    for r in results:
        annotated_frame = r.plot()  # Draw boxes and labels
        cv2.imshow("YOLO Live Detection", annotated_frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
