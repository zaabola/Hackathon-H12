from ultralytics import YOLO
import cv2
import os
import sys

# Load all three models
helmet_model = YOLO("models/final/helmet_best_yolov8s.pt")
mask_model = YOLO("runs/detect/models/finetune/mask_kaggle_finetune/weights/best.pt")
gasmask_model = YOLO("models/final/gasmask_best_yolov8s.pt")

# Open camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Could not open camera")
    exit(1)

print("📷 Starting real-time PPE detection from camera...")
print("Press 'q' to quit")

def process_frame(frame):
    """Process a single frame with all three models"""
    output = frame.copy()
    
    # Run all predictions
    helmet_result = helmet_model(frame)[0]
    mask_result = mask_model(frame)[0]
    gasmask_result = gasmask_model(frame)[0]

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
    has_helmet = len(helmet_hats) > 0
    has_good_mask = any(cls_id == 0 for cls_id in mask_detections)
    is_safe = has_helmet and (has_good_mask or has_gasmask)
    
    return output, helmet_persons, helmet_hats, mask_detections, has_gasmask, is_safe

# Real-time camera detection loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to read frame")
        break
    
    # Resize frame for faster processing
    frame = cv2.resize(frame, (640, 480))
    
    output, helmet_persons, helmet_hats, mask_detections, has_gasmask, is_safe = process_frame(frame)
    
    # Display real-time stats on frame
    stats_text = f"Helmets: {len(helmet_hats)} | Masks: {len(mask_detections)} | Gas Mask: {has_gasmask}"
    safety_text = "✅ SAFE" if is_safe else "❌ UNSAFE"
    safety_color = (0, 255, 0) if is_safe else (0, 0, 255)
    
    cv2.putText(output, stats_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(output, safety_text, (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, safety_color, 3)
    
    cv2.imshow("PPE Detection - Real-time Camera", output)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("❌ Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Camera closed")
