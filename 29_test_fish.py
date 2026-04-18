from ultralytics import YOLO
import cv2
import os

# -----------------------------
# Load trained model
# -----------------------------
model = YOLO("runs/detect/models/yolo/yolov8s_fish_base/weights/best.pt")

# -----------------------------
# Image path
# -----------------------------
img_path = "datasets/fish_dataset_fine_tune/test/images/images-2022-01-23T202324-687_jpg.rf.ff90f54e2de6a874436e9d1d7d3125c0.jpg"

# Read image
image = cv2.imread(img_path)

if image is None:
    raise FileNotFoundError(f"❌ Could not load image: {img_path}")

# -----------------------------
# Run prediction
# -----------------------------
results = model(img_path)
result = results[0]
boxes = result.boxes

# -----------------------------
# Store detections
# -----------------------------
fishes = []

for box in boxes:
    conf = float(box.conf[0])
    x1, y1, x2, y2 = map(int, box.xyxy[0])

    if conf < 0.3:   # adjust if needed
        continue

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
output_path = "outputs/predictions/fish_prediction.jpg"
cv2.imwrite(output_path, image)

print(f"✅ Prediction saved to: {output_path}")
print(f"Detected fish: {len(fishes)}")

# -----------------------------
# Show result
# -----------------------------
cv2.imshow("Fish Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()