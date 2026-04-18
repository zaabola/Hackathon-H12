"""
Fine-tune Fish Detection Model v3 — using the fishfish Roboflow dataset.

Dataset: 3,600 images (3,123 train / ~300 val / ~150 test)
Format:  YOLO Darknet (class 0 = Fish), 416x416 with augmentation
Source:  Roboflow Fish-Detection v1 (July 2022, MIT license)

This script fine-tunes the best existing fish model on the new dataset
to improve detection accuracy.
"""

from ultralytics import YOLO
import os
import sys
import torch
import multiprocessing

if __name__ == '__main__':
    multiprocessing.freeze_support()

    # ─────────────────────────────────────────────
    # GPU check
    # ─────────────────────────────────────────────
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        vram_gb = gpu.total_memory / 1e9
        print(f"GPU: {gpu.name} ({vram_gb:.1f} GB VRAM)")
    else:
        print("WARNING: No GPU detected — training will be very slow!")
        vram_gb = 0

    # ─────────────────────────────────────────────
    # Find best existing model to fine-tune from
    # ─────────────────────────────────────────────
    MODEL_PATHS = [
        "runs/detect/models/yolo/yolov8m_fish_v2/weights/best.pt",    # v2 model
        "models/yolo/yolov8m_fish_v2/weights/best.pt",                 # v2 alt
        "runs/detect/models/yolo/yolov8s_fish_merged/weights/best.pt", # v1 model
        "yolov8s.pt",                                                   # pretrained fallback
    ]

    model_path = None
    for path in MODEL_PATHS:
        if os.path.exists(path):
            model_path = path
            break

    if model_path is None:
        print("No existing model found — starting from pretrained yolov8s.pt")
        model_path = "yolov8s.pt"

    print(f"Base model: {model_path}")
    model = YOLO(model_path)

    # ─────────────────────────────────────────────
    # Training config (tuned for RTX 3050 Ti 4GB)
    # ─────────────────────────────────────────────
    DATASET_YAML = "datasets/fishfish/data.yaml"

    # Adjust batch size based on VRAM
    if vram_gb >= 8:
        BATCH_SIZE = 16
        IMGSZ = 640
    elif vram_gb >= 4:
        BATCH_SIZE = 8
        IMGSZ = 416   # match dataset resolution
    else:
        BATCH_SIZE = 4
        IMGSZ = 416

    print(f"Training config: batch={BATCH_SIZE}, imgsz={IMGSZ}")

    # ─────────────────────────────────────────────
    # Train
    # ─────────────────────────────────────────────
    results = model.train(
        data=DATASET_YAML,
        epochs=50,
        batch=BATCH_SIZE,
        imgsz=IMGSZ,
        
        # Learning rate — lower for fine-tuning
        lr0=0.001,
        lrf=0.01,
        
        # Augmentation
        mosaic=1.0,
        flipud=0.5,
        fliplr=0.5,
        
        # Regularization
        weight_decay=0.0005,
        
        # Output
        project="runs/detect/models/yolo",
        name="yolov8_fish_v3",
        
        # Save best + last
        save=True,
        save_period=10,
        
        # Early stopping
        patience=15,
        
        # Workers — 0 to avoid Windows multiprocessing issues
        workers=0,
        
        # Verbose
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("  Training Complete!")
    print("=" * 60)
    print(f"  Best model: runs/detect/models/yolo/yolov8_fish_v3/weights/best.pt")
    print(f"\nNext steps:")
    print(f"  1. Test on video: python scripts/38_test_video_v2.py")
    print(f"  2. Update model path in test scripts to use v3 weights")
