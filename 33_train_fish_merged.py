from ultralytics import YOLO
import torch

def main():
    print("GPU available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    # Start from pretrained YOLOv8s (COCO weights)
    model = YOLO("yolov8s.pt")

    results = model.train(
        data="datasets/fish_merged/data.yaml",
        epochs=60,
        imgsz=640,
        batch=8,
        device=0,
        workers=2,
        amp=True,
        cache=False,
        
        # Single class = all fish treated as "fish"
        single_cls=True,
        
        # Better augmentation for underwater scenes
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.1,
        hsv_h=0.02,        # underwater color shifts
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=15.0,       # fish can be rotated
        flipud=0.2,         # fish can be upside down
        fliplr=0.5,
        scale=0.5,
        translate=0.1,
        
        # Output
        project="models/yolo",
        name="yolov8s_fish_merged",
        exist_ok=True,
        patience=15,        # early stopping if no improvement for 15 epochs
    )

if __name__ == "__main__":
    main()
