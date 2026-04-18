from ultralytics import YOLO
import torch
import os
import shutil

def main():
    print("GPU available:", torch.cuda.is_available())
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

    # -----------------------------
    # LOAD YOUR BEST HELMET MODEL
    # -----------------------------
    model = YOLO("models/final/helmet_best_yolov8s.pt")

    print("\n🔥 Starting SAFE fine-tuning on helmet dataset...\n")

    results = model.train(
        data="datasets/hellmet_fine_tuned_dataset/data.yaml",

        # -----------------------------
        # SAFE FINETUNE SETTINGS
        # -----------------------------
        epochs=10,
        imgsz=640,
        batch=8,
        device=0,
        workers=0,

        # IMPORTANT → don't destroy learned weights
        lr0=0.0001,
        lrf=0.01,
        freeze=10,
        patience=5,
        pretrained=True,

        project="models/finetune",
        name="helmet_yolov8s_finetuned_SAFE",
        exist_ok=True
    )

    # -----------------------------
    # SAVE AS NEW FILE
    # -----------------------------
    os.makedirs("models/final", exist_ok=True)

    possible_paths = [
        "models/finetune/helmet_yolov8s_finetuned_SAFE/weights/best.pt",
        "runs/detect/helmet_yolov8s_finetuned_SAFE/weights/best.pt"
    ]

    for src in possible_paths:
        if os.path.exists(src):
            dst = "models/final/helmet_best_yolov8s_finetuned.pt"
            shutil.copy(src, dst)
            print(f"\n✅ Fine-tuned helmet model saved to: {dst}")
            return

    print("⚠️ Could not auto-copy. Check runs/detect manually.")

if __name__ == "__main__":
    main()