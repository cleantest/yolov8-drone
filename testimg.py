from ultralytics import YOLO
import cv2

# Load your trained model
model = YOLO('runs/detect/train/weights/best.pt')

# Path to your test image
image_path = 'test\images\-1134_png_jpg.rf.83c42ea47fb5f0cba8ee6f6e7847b81a.jpg'  # <-- change to your image path

# Read the image
img = cv2.imread(image_path)

# Run inference
results = model(img)

# Visualize results
for r in results:
    annotated_img = r.plot()  # draw boxes and labels

# Display the result
cv2.imshow("YOLO Image Detection", annotated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
