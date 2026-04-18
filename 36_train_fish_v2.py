"""
Improved fish detection training script (v2).
Key improvements over 33_train_fish_merged.py:
  - YOLOv8m (25.9M params) instead of YOLOv8s (11.2M)
  - 100 epochs with cosine LR schedule
  - Dropout regularization to reduce false positives
  - Better augmentation for underwater conditions
  - Later mosaic disable for precise box learning

Tuned for RTX 3050 Ti (4GB VRAM):
  - imgsz=640, batch=2 to fit in 4GB
  - amp=True for mixed precision (saves ~40% VRAM)
"""

from ultralytics import YOLO
import torch


def main():
    print("=" * 60)
    print("  Fish Detection Training v2 (Improved)")
    print("  Optimized for RTX 3050 Ti (4GB VRAM)")
    print("=" * 60)
    print(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name} ({vram_gb:.1f} GB)")

    # ─────────────────────────────────────────────
    # Use YOLOv8m — 2.3x more parameters than v8s
    # Much better feature extraction for small objects
    # Fits in 4GB VRAM with batch=2 + amp=True
    # ─────────────────────────────────────────────
    model = YOLO("yolov8m.pt")

    results = model.train(
        data="datasets/fish_merged/data.yaml",

        # ── Training duration ──
        epochs=100,              # v1 was 60 — losses were still dropping
        patience=20,             # early stopping if no improvement for 20 epochs

        # ── Resolution & batch (tuned for 4GB VRAM) ──
        imgsz=640,               # keep at 640 to fit in 4GB VRAM
        batch=2,                 # small batch for 4GB GPU
        # If you STILL get OOM, add: cache=False, workers=0

        # ── Hardware ──
        device=0,
        workers=2,
        amp=True,                # CRITICAL for 4GB — mixed precision saves ~40% VRAM
        cache=False,

        # ── Single class ──
        single_cls=True,         # all fish = one class

        # ── Learning rate ──
        cos_lr=True,             # cosine annealing — smoother convergence
        lr0=0.01,                # initial learning rate
        lrf=0.01,                # final LR = lr0 * lrf

        # ── Regularization ──
        dropout=0.1,             # helps reduce false positives on coral/rocks

        # ── Augmentation (tuned for underwater scenes) ──
        mosaic=1.0,
        close_mosaic=20,         # v1 was 10 — keep mosaic longer, then fine-tune
        mixup=0.15,
        copy_paste=0.1,

        # Color augmentation (underwater-specific)
        hsv_h=0.03,             # v1 was 0.02 — broader hue shifts for water color
        hsv_s=0.8,              # v1 was 0.7 — underwater saturation varies a lot
        hsv_v=0.5,              # v1 was 0.4 — brightness varies dramatically

        # Geometric augmentation
        degrees=15.0,           # fish can be rotated
        flipud=0.2,             # fish can be upside down
        fliplr=0.5,
        scale=0.5,
        translate=0.1,
        perspective=0.001,      # slight perspective distortion (new)
        erasing=0.3,            # random erasing to simulate occlusion

        # ── Output ──
        project="models/yolo",
        name="yolov8m_fish_v2",
        exist_ok=True,
    )

    print("\n" + "=" * 60)
    print("  Training Complete!")
    print("=" * 60)
    print("Model saved to: models/yolo/yolov8m_fish_v2/weights/best.pt")
    print("\nNext step: test on video with:")
    print("  python scripts/38_test_video_v2.py")


if __name__ == "__main__":
    main()
