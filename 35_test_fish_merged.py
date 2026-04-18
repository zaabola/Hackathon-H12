from ultralytics import YOLO
import cv2
import os

# -----------------------------
# Load the NEW merged model
# -----------------------------
model = YOLO("runs/detect/models/yolo/yolov8m_fish_v2/weights/best.pt")

# -----------------------------
# Image path
# -----------------------------
img_path = "datasets/fish_dataset/test/images/41_png.rf.6ec5e46b9002e467174e8f16035c57fe.jpg"

# Read image
image = cv2.imread(img_path)

if image is None:
    raise FileNotFoundError(f"Could not load image: {img_path}")

# -----------------------------
# Run prediction
# -----------------------------
results = model(img_path, conf=0.3)
result = results[0]
boxes = result.boxes

# -----------------------------
# Store detections
# -----------------------------
fishes = []

for box in boxes:
    conf = float(box.conf[0])
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    fishes.append((x1, y1, x2, y2, conf))

# -----------------------------
# Draw fish boxes (GREEN)
# -----------------------------
for (x1, y1, x2, y2, conf) in fishes:
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image, f"Fish {conf:.2f}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# -----------------------------
# Save result
# -----------------------------
os.makedirs("outputs/predictions", exist_ok=True)
output_path = "outputs/predictions/fish_prediction_merged.jpg"
cv2.imwrite(output_path, image)

print(f"Prediction saved to: {output_path}")
print(f"Detected fish: {len(fishes)}")

# -----------------------------
# Show result
# -----------------------------
cv2.imshow("Fish Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
