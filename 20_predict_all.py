from ultralytics import YOLO
import cv2
import os
import sys

# Accept image path from command line or use default
if len(sys.argv) > 1:
    img_path = sys.argv[1]
else:
    img_path = "C:/Users/zaabola/Desktop/PPE_Project/datasets/gas_mask_dataset/test/images/frame_0963_jpg.rf.9069743f4fad8a6dcf2fa31cd4a1b633.jpg"

# Load all three models
helmet_model = YOLO("models/final/helmet_best_yolov8s.pt")
mask_model = YOLO("models/final/mask_best_yolov8s.pt")
gasmask_model = YOLO("models/final/gasmask_best_yolov8s.pt")

# Read image
image = cv2.imread(img_path)
if image is None:
    raise FileNotFoundError(f"❌ Image not found: {img_path}")

output = image.copy()

# Run all predictions
helmet_result = helmet_model(img_path)[0]
mask_result = mask_model(img_path)[0]
gasmask_result = gasmask_model(img_path)[0]

# ==================== HELMET DETECTION ====================
helmet_persons = []
helmet_hats = []

for box in helmet_result.boxes:
    cls_id = int(box.cls[0])
    conf = float(box.conf[0])
    x1, y1, x2, y2 = map(int, box.xyxy[0])

    if conf < 0.2:
        continue

    if cls_id == 1:  # person
        helmet_persons.append((x1, y1, x2, y2, conf))
    elif cls_id == 0:  # hat
        helmet_hats.append((x1, y1, x2, y2, conf))

# Draw helmet hats (BLUE)
for (hx1, hy1, hx2, hy2, hconf) in helmet_hats:
    cv2.rectangle(output, (hx1, hy1), (hx2, hy2), (255, 0, 0), 2)
    cv2.putText(output, f"Hat {hconf:.2f}", (hx1, hy1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

# Match hats to persons and draw
for (px1, py1, px2, py2, pconf) in helmet_persons:
    has_helmet = False

    for (hx1, hy1, hx2, hy2, hconf) in helmet_hats:
        hat_center_x = (hx1 + hx2) // 2
        hat_center_y = (hy1 + hy2) // 2

        if px1 <= hat_center_x <= px2 and py1 <= hat_center_y <= py1 + (py2 - py1) * 0.4:
            has_helmet = True
            break

    if has_helmet:
        color = (0, 255, 0)  # Green
        label = f"Helmet {pconf:.2f}"
    else:
        color = (0, 0, 255)  # Red
        label = f"No Helmet {pconf:.2f}"

    cv2.rectangle(output, (px1, py1), (px2, py2), color, 2)
    cv2.putText(output, label, (px1, py1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

# ==================== MASK DETECTION ====================
class_names_mask = {
    0: "With Mask",
    1: "Without Mask",
    2: "Incorrect Mask"
}

mask_detections = []
for box in mask_result.boxes:
    cls_id = int(box.cls[0])
    conf = float(box.conf[0])
    x1, y1, x2, y2 = map(int, box.xyxy[0])

    if conf < 0.3:
        continue

    mask_detections.append(cls_id)
    label = f"{class_names_mask.get(cls_id, 'Unknown')} {conf:.2f}"

    if cls_id == 0:
        color = (0, 255, 0)     # Green
    elif cls_id == 1:
        color = (0, 0, 255)     # Red
    else:
        color = (0, 165, 255)   # Orange

    cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
    cv2.putText(output, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

# ==================== GAS MASK DETECTION ====================
class_names_gas = {
    0: "Oxygen_tube",
    1: "GasMask",
    2: "Person"
}

has_gasmask = False
for box in gasmask_result.boxes:
    cls_id = int(box.cls[0])
    conf = float(box.conf[0])
    x1, y1, x2, y2 = map(int, box.xyxy[0])

    if conf < 0.25:
        continue

    if cls_id == 1:  # gasmask
        has_gasmask = True

    label = f"{class_names_gas.get(cls_id, 'Unknown')} {conf:.2f}"

    if cls_id == 0:      # Oxygen_tube
        color = (255, 255, 0)   # Cyan
    elif cls_id == 1:    # gasmask
        color = (0, 255, 0)     # Green
    elif cls_id == 2:    # person
        color = (0, 0, 255)     # Red
    else:
        color = (255, 255, 255)

    cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
    cv2.putText(output, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

# ==================== SAFETY ANALYSIS ====================
# Determine safety status
has_helmet = len(helmet_hats) > 0
has_good_mask = any(cls_id == 0 for cls_id in mask_detections)

is_safe = has_helmet and (has_good_mask or has_gasmask)

# Save and display
os.makedirs("outputs/predictions", exist_ok=True)
output_path = "outputs/predictions/all_ppe_detection.jpg"
cv2.imwrite(output_path, output)

print(f"✅ Result saved to: {output_path}")
print(f"🎖️  Helmets: {len(helmet_persons)} persons, {len(helmet_hats)} hats")
print(f"😷 Masks: {len(mask_detections)} (With Mask: {sum(1 for m in mask_detections if m == 0)})")
print(f"🥽 Gas Masks: {has_gasmask}")
print(f"\n{'='*50}")
if is_safe:
    print(f"✅ STATUS: SAFE")
else:
    print(f"❌ STATUS: UNSAFE")
print(f"{'='*50}\n")

cv2.imshow("PPE Detection - Combined", output)
cv2.waitKey(0)
cv2.destroyAllWindows()