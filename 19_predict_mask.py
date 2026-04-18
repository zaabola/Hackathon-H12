from ultralytics import YOLO
import cv2
import os

# -----------------------------
# Load trained model
# -----------------------------
model = YOLO("runs/detect/models/finetune/mask_kaggle_finetune/weights/best.pt")

# -----------------------------
# Image path (CHANGE THIS)
# -----------------------------
img_path = "datasets/mask_yolo_finetune/train/images/maksssksksss3.png"  # change this

# Load image
image = cv2.imread(img_path)

if image is None:
    raise FileNotFoundError(f"❌ Image not found: {img_path}")

# Run prediction
results = model(img_path)
result = results[0]
boxes = result.boxes

# -----------------------------
# Class names
# -----------------------------
class_names = {
    0: "With Mask",
    1: "Without Mask",
    2: "Incorrect Mask"
}

# -----------------------------
# Draw detections
# -----------------------------
for box in boxes:
    cls_id = int(box.cls[0])
    conf = float(box.conf[0])
    x1, y1, x2, y2 = map(int, box.xyxy[0])

    if conf < 0.3:
        continue

    label = f"{class_names.get(cls_id, 'Unknown')} {conf:.2f}"

    # Colors
    if cls_id == 0:
        color = (0, 255, 0)     # Green
    elif cls_id == 1:
        color = (0, 0, 255)     # Red
    else:
        color = (0, 165, 255)   # Orange

    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(image, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

# -----------------------------
# Save output
# -----------------------------
os.makedirs("outputs/predictions", exist_ok=True)

output_path = "outputs/predictions/mask_prediction.jpg"
cv2.imwrite(output_path, image)

print(f"✅ Prediction saved to: {output_path}")

# Show image
cv2.imshow("Mask Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()