from ultralytics import YOLO
import cv2
import os

# -----------------------------
# IMAGE TO TEST
# -----------------------------
img_path = "datasets/helmet_yolo_clean/test/images/11.jpg"

image = cv2.imread(img_path)
if image is None:
    raise FileNotFoundError(f"Image not found: {img_path}")

# -----------------------------
# LOAD MODELS
# -----------------------------
old_helmet = YOLO("models/final/helmet_best_yolov8s.pt")
new_helmet = YOLO("models/final/helmet_best_yolov8s_finetuned.pt")

old_mask = YOLO("models/final/mask_best_yolov8s.pt")
new_mask = YOLO("runs/detect/models/finetune/mask_kaggle_finetune/weights/best.pt")

old_gasmask = YOLO("models/final/gasmask_best_yolov8s.pt")
new_gasmask = YOLO("models/final/gasmask_best_yolov8s_finetuned.pt")

# -----------------------------
# RUN INFERENCE
# -----------------------------
models = {
    "OLD_HELMET": old_helmet,
    "NEW_HELMET": new_helmet,
    "OLD_MASK": old_mask,
    "NEW_MASK": new_mask,
    "OLD_GASMASK": old_gasmask,
    "NEW_GASMASK": new_gasmask
}

for name, model in models.items():
    result = model(img_path)[0]
    print(f"\n===== {name} =====")
    print(f"Detections: {len(result.boxes)}")

    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        print(f"class={cls_id}, conf={conf:.2f}, box=({x1},{y1},{x2},{y2})")