from ultralytics import YOLO
import torch

def main():
    print("GPU available:", torch.cuda.is_available())
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

    model = YOLO("yolov8m.pt")

    results = model.train(
        data="datasets/gas_mask_dataset/data.yaml",
        epochs=10,
        imgsz=640,
        batch=4,
        device=0,
        workers=0,
        project="models/yolo",
        name="yolov8m_gas_mask_test",
        exist_ok=True
    )

if __name__ == "__main__":
    main()