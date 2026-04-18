from ultralytics import YOLO
import torch
import os
import shutil

def main():
    print("====================================")
    print(" MASK MODEL SUPERVISED FINE-TUNING ")
    print("====================================")

    print("GPU available:", torch.cuda.is_available())
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

    # ==========================================
    # LOAD YOUR EXISTING BEST MASK MODEL
    # ==========================================
    model = YOLO("models/final/mask_best_yolov8s.pt")

    print("\n🚀 Starting supervised fine-tuning on NEW Kaggle mask dataset...\n")

    results = model.train(
        data="datasets/mask_yolo_finetune/data.yaml",

        # Fine-tuning settings
        epochs=15,
        imgsz=640,
        batch=8,
        device=0,
        workers=0,

        # Save separately
        project="models/finetune",
        name="mask_kaggle_finetune",
        exist_ok=True,

        # Stabilize training
        patience=5,
        lr0=0.0005,      # smaller LR for fine-tuning
        lrf=0.01,
        cos_lr=True,

        # Augmentation (light, not crazy)
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
        degrees=5.0,
        translate=0.05,
        scale=0.2,
        fliplr=0.5,
        mosaic=0.2,

        # Optional
        verbose=True
    )

    # ==========================================
    # SAVE FINE-TUNED MODEL
    # ==========================================
    os.makedirs("models/final", exist_ok=True)

    possible_paths = [
        "models/finetune/mask_kaggle_finetune/weights/best.pt",
        "runs/detect/models/finetune/mask_kaggle_finetune/weights/best.pt",
        "runs/detect/mask_kaggle_finetune/weights/best.pt"
    ]

    saved = False
    for src in possible_paths:
        if os.path.exists(src):
            dst = "models/final/mask_best_yolov8s_finetuned.pt"
            shutil.copy(src, dst)
            print(f"\n✅ Fine-tuned model saved to: {dst}")
            saved = True
            break

    if not saved:
        print("\n⚠️ Could not auto-copy best.pt")
        print("Check manually inside your training folder.")

if __name__ == "__main__":
    main()