from ultralytics import YOLO
import cv2
import os

# -----------------------------
# Load trained gas mask model
# -----------------------------
model = YOLO("models/final/gasmask_best_yolov8s.pt")

# -----------------------------
# Image path (CHANGE THIS)
# -----------------------------
img_path = "datasets/gas_mask_dataset/test/images/frame_0002_jpg.rf.d8caaefc84d273422b5b6ff35af31a1a.jpg"  # <-- change this

# Read image
image = cv2.imread(img_path)

if image is None:
    raise FileNotFoundError(f"❌ Could not load image: {img_path}")

# Run prediction
results = model(img_path)
result = results[0]
boxes = result.boxes

# -----------------------------
# Class names
# -----------------------------
class_names = {
    0: "Oxygen_tube",
    1: "GasMask",
    2: "Person"
}

# -----------------------------
# Draw detections
# -----------------------------
for box in boxes:
    cls_id = int(box.cls[0])
    conf = float(box.conf[0])
    x1, y1, x2, y2 = map(int, box.xyxy[0])

    if conf < 0.25:
        continue

    label = f"{class_names.get(cls_id, 'Unknown')} {conf:.2f}"

    # Colors by class
    if cls_id == 0:      # Oxygen_tube
        color = (255, 255, 0)   # cyan-ish
    elif cls_id == 1:    # gasmask
        color = (0, 255, 0)     # green
    elif cls_id == 2:    # person
        color = (0, 0, 255)     # red
    else:
        color = (255, 255, 255)

    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(image, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

# -----------------------------
# Save result
# -----------------------------
os.makedirs("outputs/predictions", exist_ok=True)
output_path = "outputs/predictions/gasmask_prediction.jpg"
cv2.imwrite(output_path, image)

print(f"✅ Prediction saved to: {output_path}")

# Show result
cv2.imshow("Gas Mask Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()