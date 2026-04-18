"""
Improved video fish detection script (v2).
Key improvements over 34_test_video_fish_merged.py:
  - Uses model.track() instead of model() for persistent tracking across frames
  - Higher inference resolution (1280px)
  - Lower confidence threshold (0.25) for better recall
  - Test-time augmentation (TTA) option
  - Smoother bounding boxes via tracking
  - Fish ID tracking with unique colors
  - Falls back to v1 model if v2 not trained yet
"""

from ultralytics import YOLO
import cv2
import os
import numpy as np
import sys

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
USE_TRACKING = True          # Use ByteTrack for persistent fish tracking
USE_TTA = False              # Test-time augmentation (slower but more robust)
CONF_THRESHOLD = 0.25        # Lower than v1 (0.35) — tracking smooths false positives
IOU_THRESHOLD = 0.45
IMGSZ = 640                  # Match training resolution (4GB VRAM safe)
SHOW_IDS = True              # Show fish track IDs on video

# ─────────────────────────────────────────────
# Load model (try v2 first, fall back to v1)
# ─────────────────────────────────────────────
MODEL_PATHS = [
    "runs/detect/models/yolo/yolov8_fish_v3/weights/best.pt",    # v3 model
    "runs/detect/models/yolo/yolov8m_fish_v2/weights/best.pt",    # v2 model
    "models/yolo/yolov8m_fish_v2/weights/best.pt",                 # v2 alt path
    "runs/detect/models/yolo/yolov8s_fish_merged/weights/best.pt", # v1 fallback
]

model_path = None
for path in MODEL_PATHS:
    if os.path.exists(path):
        model_path = path
        break

if model_path is None:
    print("ERROR: No model found. Train first with:")
    print("  python scripts/36_train_fish_v2.py")
    sys.exit(1)

print(f"Loading model: {model_path}")
model = YOLO(model_path)

# ─────────────────────────────────────────────
# Video path
# ─────────────────────────────────────────────
video_path = "datasets/fish_dataset/test/images/videoplayback (1).mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise FileNotFoundError(f"Could not open video: {video_path}")

# ─────────────────────────────────────────────
# Output setup
# ─────────────────────────────────────────────
os.makedirs("outputs/videos", exist_ok=True)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

output_path = "outputs/videos/fish_output_v2.mp4"

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

print(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")
print(f"Settings: tracking={USE_TRACKING}, TTA={USE_TTA}, conf={CONF_THRESHOLD}, imgsz={IMGSZ}")

# ─────────────────────────────────────────────
# Color palette for tracked fish IDs
# ─────────────────────────────────────────────
def get_color(track_id):
    """Generate a unique, vibrant color for each fish track ID."""
    np.random.seed(int(track_id) * 7 + 13)
    hue = np.random.randint(0, 180)
    color_hsv = np.uint8([[[hue, 255, 230]]])
    color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
    return tuple(int(c) for c in color_bgr)

# ─────────────────────────────────────────────
# Process video
# ─────────────────────────────────────────────
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # ── Run inference ──
    if USE_TRACKING:
        results = model.track(
            frame,
            imgsz=IMGSZ,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            persist=True,                # maintain tracks across frames
            tracker="bytetrack.yaml",    # ByteTrack is good for crowded scenes
            verbose=False,
        )
    else:
        results = model(
            frame,
            imgsz=IMGSZ,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            augment=USE_TTA,
            verbose=False,
        )

    result = results[0]
    boxes = result.boxes

    # ── Draw detections ──
    fish_count = 0
    for box in boxes:
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Get track ID if tracking is enabled
        track_id = None
        if USE_TRACKING and box.id is not None:
            track_id = int(box.id[0])

        # Choose color: unique per ID if tracking, green otherwise
        if track_id is not None:
            color = get_color(track_id)
            label = f"Fish #{track_id} {conf:.2f}"
        else:
            color = (0, 255, 0)
            label = f"Fish {conf:.2f}"

        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Draw label with background
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
        )
        cv2.rectangle(
            frame,
            (x1, y1 - label_h - baseline - 4),
            (x1 + label_w, y1),
            color,
            -1,
        )
        cv2.putText(
            frame, label, (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2,
        )

        fish_count += 1

    # ── Display info overlay ──
    info_text = f"Fish Count: {fish_count}"
    cv2.putText(frame, info_text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Progress bar
    progress = frame_count / max(total_frames, 1)
    bar_width = 200
    cv2.rectangle(frame, (20, height - 30), (20 + bar_width, height - 15), (80, 80, 80), -1)
    cv2.rectangle(frame, (20, height - 30), (20 + int(bar_width * progress), height - 15),
                  (0, 255, 255), -1)
    cv2.putText(frame, f"{progress*100:.0f}%", (230, height - 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Write frame
    out.write(frame)

    # Show frame
    cv2.imshow("Fish Detection v2", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

    if frame_count % 100 == 0:
        print(f"Processed {frame_count}/{total_frames} frames ({progress*100:.0f}%)")

# ─────────────────────────────────────────────
# Cleanup
# ─────────────────────────────────────────────
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"\nVideo saved to: {output_path}")
print(f"Total frames processed: {frame_count}")
