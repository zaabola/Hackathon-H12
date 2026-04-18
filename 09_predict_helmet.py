from ultralytics import YOLO
import cv2
import os

# -----------------------------
# Load trained model
# -----------------------------
model = YOLO("models/final/helmet_best_yolov8s.pt")

# -----------------------------
# Image path
# -----------------------------
img_path = "datasets/helmet_yolo_clean/test/images/11.jpg"

# Read image
image = cv2.imread(img_path)

if image is None:
    raise FileNotFoundError(f"❌ Could not load image: {img_path}")

# Run prediction
results = model(img_path)
result = results[0]
boxes = result.boxes

# -----------------------------
# Separate detections
# -----------------------------
persons = []
hats = []

for box in boxes:
    cls_id = int(box.cls[0])
    conf = float(box.conf[0])
    x1, y1, x2, y2 = map(int, box.xyxy[0])

    if conf < 0.2:   # lowered threshold for debugging
        continue

    if cls_id == 1:  # person
        persons.append((x1, y1, x2, y2, conf))
    elif cls_id == 0:  # hat
        hats.append((x1, y1, x2, y2, conf))

# -----------------------------
# Draw all hat boxes (BLUE)
# -----------------------------
for (hx1, hy1, hx2, hy2, hconf) in hats:
    cv2.rectangle(image, (hx1, hy1), (hx2, hy2), (255, 0, 0), 2)
    cv2.putText(image, f"Hat {hconf:.2f}", (hx1, hy1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

# -----------------------------
# Match hats to persons
# -----------------------------
for (px1, py1, px2, py2, pconf) in persons:
    has_helmet = False

    for (hx1, hy1, hx2, hy2, hconf) in hats:
        hat_center_x = (hx1 + hx2) // 2
        hat_center_y = (hy1 + hy2) // 2

        # Hat should be in upper part of person
        if px1 <= hat_center_x <= px2 and py1 <= hat_center_y <= py1 + (py2 - py1) * 0.4:
            has_helmet = True
            break

    if has_helmet:
        color = (0, 255, 0)  # Green
        label = f"Helmet {pconf:.2f}"
    else:
        color = (0, 0, 255)  # Red
        label = f"No Helmet {pconf:.2f}"

    cv2.rectangle(image, (px1, py1), (px2, py2), color, 2)
    cv2.putText(image, label, (px1, py1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

# -----------------------------
# Save result
# -----------------------------
os.makedirs("outputs/predictions", exist_ok=True)
output_path = "outputs/predictions/helmet_prediction.jpg"
cv2.imwrite(output_path, image)

print(f"✅ Prediction saved to: {output_path}")
print(f"Detected persons: {len(persons)}")
print(f"Detected hats: {len(hats)}")

# Show result
cv2.imshow("Helmet Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()