from ultralytics import YOLO
import torch

def main():
    print("GPU available:", torch.cuda.is_available())
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

    # 🔥 LOAD YOUR TRAINED MODEL (NOT yolov8s.pt)
    model = YOLO("runs/detect/models/yolo/yolov8s_fish_base/weights/best.pt")

    # Fine-tune on second dataset
    results = model.train(
        data="datasets/fish_dataset_fine_tune/data.yaml",  # 🔥 second dataset
        epochs=30,                                         # less epochs for finetune
        imgsz=640,
        batch=8,
        device=0,
        workers=2,
        amp=True,
        project="models/yolo",
        name="yolov8s_fish_finetuned",
        exist_ok=True
    )

if __name__ == "__main__":
    main()