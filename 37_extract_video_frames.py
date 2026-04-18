"""
Extract frames from video and generate pseudo-labels using the current model.
This bridges the 'domain gap' between training images and the actual video footage.

Workflow:
  1. Extract every N-th frame from the video
  2. Run the current model to auto-generate labels (pseudo-labels)
  3. Save frames + labels in YOLO format
  4. Manually review/correct labels (optional but recommended)  
  5. Add to training set and retrain

Usage:
  python scripts/37_extract_video_frames.py
  
Then review the outputs in datasets/fish_video_frames/ and add good ones to
datasets/fish_merged/train/
"""

from ultralytics import YOLO
import cv2
import os
import sys

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
FRAME_INTERVAL = 30          # Extract every N-th frame (30 = ~1 per second at 30fps)
CONF_THRESHOLD = 0.20        # Low threshold to catch more fish (review manually)
IMGSZ = 1280                 # High resolution for better detection
MIN_FISH_PER_FRAME = 1       # Only keep frames with at least this many detections

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
VIDEO_PATH = "datasets/fish_dataset/test/images/11.mp4"
OUTPUT_DIR = "datasets/fish_video_frames"

# Try to find the best available model
MODEL_PATHS = [
    "runs/detect/models/yolo/yolov8m_fish_v2/weights/best.pt",
    "models/yolo/yolov8m_fish_v2/weights/best.pt",
    "runs/detect/models/yolo/yolov8s_fish_merged/weights/best.pt",
]

model_path = None
for path in MODEL_PATHS:
    if os.path.exists(path):
        model_path = path
        break

if model_path is None:
    print("ERROR: No model found.")
    sys.exit(1)

print(f"Using model: {model_path}")
model = YOLO(model_path)

# ─────────────────────────────────────────────
# Setup output dirs
# ─────────────────────────────────────────────
img_dir = os.path.join(OUTPUT_DIR, "images")
lbl_dir = os.path.join(OUTPUT_DIR, "labels")
os.makedirs(img_dir, exist_ok=True)
os.makedirs(lbl_dir, exist_ok=True)

# ─────────────────────────────────────────────
# Open video
# ─────────────────────────────────────────────
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise FileNotFoundError(f"Could not open video: {VIDEO_PATH}")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

print(f"Video: {width}x{height} @ {fps}fps, {total_frames} total frames")
print(f"Extracting every {FRAME_INTERVAL} frames (~{total_frames // FRAME_INTERVAL} frames)")

# ─────────────────────────────────────────────
# Extract and label frames
# ─────────────────────────────────────────────
frame_count = 0
saved_count = 0
skipped_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % FRAME_INTERVAL != 0:
        continue

    # Run model
    results = model(frame, imgsz=IMGSZ, conf=CONF_THRESHOLD, verbose=False)
    result = results[0]
    boxes = result.boxes

    if len(boxes) < MIN_FISH_PER_FRAME:
        skipped_count += 1
        continue

    # Save image
    frame_name = f"video_frame_{frame_count:06d}"
    img_path = os.path.join(img_dir, f"{frame_name}.jpg")
    cv2.imwrite(img_path, frame)

    # Save YOLO-format labels (class cx cy w h, normalized)
    label_lines = []
    for box in boxes:
        # Get normalized xywh (center x, center y, width, height)
        xywhn = box.xywhn[0]  # already normalized to [0, 1]
        cx, cy, w, h = float(xywhn[0]), float(xywhn[1]), float(xywhn[2]), float(xywhn[3])
        label_lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    lbl_path = os.path.join(lbl_dir, f"{frame_name}.txt")
    with open(lbl_path, 'w') as f:
        f.write("\n".join(label_lines) + "\n")

    saved_count += 1

    if saved_count % 10 == 0:
        print(f"  Saved {saved_count} frames (at frame {frame_count}/{total_frames})")

cap.release()

print(f"\n{'=' * 60}")
print(f"  Frame Extraction Complete!")
print(f"{'=' * 60}")
print(f"  Total video frames: {total_frames}")
print(f"  Frames extracted: {saved_count}")
print(f"  Frames skipped (no fish): {skipped_count}")
print(f"  Saved to: {OUTPUT_DIR}")
print(f"\nNext steps:")
print(f"  1. Review images in {img_dir}")
print(f"  2. Check/correct labels in {lbl_dir}")
print(f"  3. Copy good frames to datasets/fish_merged/train/")
print(f"  4. Retrain with: python scripts/36_train_fish_v2.py")
