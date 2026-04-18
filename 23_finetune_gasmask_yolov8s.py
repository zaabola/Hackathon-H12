from ultralytics import YOLO
import torch
import os
import shutil

def main():
    print("GPU available:", torch.cuda.is_available())
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

    # -----------------------------
    # LOAD YOUR BEST MODEL
    # -----------------------------
    model = YOLO("models/final/gasmask_best_yolov8s.pt")

    print("\n🔥 Starting SAFE fine-tuning on gasmask dataset...\n")

    results = model.train(
        data="datasets/gas_mask_dataset/data.yaml",

        # 🔥 IMPORTANT SETTINGS (to avoid destroying model)
        epochs=10,            # small finetune
        imgsz=640,
        batch=8,
        device=0,
        workers=0,

        # VERY IMPORTANT
        lr0=0.0001,          # low learning rate (keeps knowledge)
        lrf=0.01,

        # stability
        patience=5,
        pretrained=True,
        freeze=10,           # freeze backbone (VERY IMPORTANT)

        project="models/finetune",
        name="gasmask_yolov8s_finetuned_SAFE",
        exist_ok=True
    )

    # -----------------------------
    # SAVE WITHOUT OVERWRITING
    # -----------------------------
    os.makedirs("models/final", exist_ok=True)

    possible_paths = [
        "models/finetune/gasmask_yolov8s_finetuned_SAFE/weights/best.pt",
        "runs/detect/gasmask_yolov8s_finetuned_SAFE/weights/best.pt"
    ]

    for src in possible_paths:
        if os.path.exists(src):
            dst = "models/final/gasmask_best_yolov8s_finetuned.pt"
            shutil.copy(src, dst)
            print(f"\n✅ Fine-tuned model saved to: {dst}")
            return

    print("⚠️ Could not auto-copy. Check runs/detect manually.")

if __name__ == "__main__":
    main()