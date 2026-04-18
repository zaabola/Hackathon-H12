from ultralytics import YOLO
import torch
import os
import shutil

def main():
    print("GPU available:", torch.cuda.is_available())
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

    model = YOLO("yolov8s.pt")

    results = model.train(
        data="datasets/gas_mask_dataset/data.yaml",
        epochs=30,
        imgsz=640,
        batch=8,
        device=0,
        workers=0,
        project="models/yolo",
        name="yolov8s_gasmask_final",
        exist_ok=True
    )

    # Create clean final folder
    os.makedirs("models/final", exist_ok=True)

    # Possible locations of best.pt
    possible_paths = [
        "models/yolo/yolov8s_gasmask_final/weights/best.pt",
        "runs/detect/models/yolo/yolov8s_gasmask_final/weights/best.pt",
        "runs/detect/yolov8s_gasmask_final/weights/best.pt"
    ]

    for src in possible_paths:
        if os.path.exists(src):
            dst = "models/final/gasmask_best_yolov8s.pt"
            shutil.copy(src, dst)
            print(f"✅ Final gas mask model copied to: {dst}")
            return

    print("⚠️ best.pt was not found automatically. Please copy it manually.")

if __name__ == "__main__":
    main()