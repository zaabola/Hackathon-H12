"""
Stable Video Fish Detection — smooth bounding boxes & labels.

Key improvements over 38_test_video_v2.py:
  - Exponential Moving Average (EMA) smoothing on bounding box coordinates
  - Confidence smoothing across frames (no flickering labels)
  - Tracks persist longer before being dropped (handles brief occlusions)
  - Stable label text (no rapid conf% changes)
  - Uses the best v3/v4 model
"""

from ultralytics import YOLO
import cv2
import os
import numpy as np
import sys
import multiprocessing
from collections import defaultdict

if __name__ == '__main__':
    multiprocessing.freeze_support()

    # ─────────────────────────────────────────────
    # Configuration
    # ─────────────────────────────────────────────
    CONF_THRESHOLD = 0.25
    IOU_THRESHOLD = 0.45
    IMGSZ = 416              # match training resolution

    # Smoothing parameters
    ALPHA = 0.4               # EMA factor: lower = smoother but laggier (0.2-0.5 is good)
    CONF_ALPHA = 0.3          # Confidence smoothing factor
    TRACK_TIMEOUT = 15        # Frames to keep showing a track after it disappears

    # ─────────────────────────────────────────────
    # Load model
    # ─────────────────────────────────────────────
    MODEL_PATHS = [
        "runs/detect/models/yolo/yolov8m_finetunefinetune_with_bg/weights/best.pt",
        "runs/detect/models/yolo/yolov8m_finetunefinetune_with_bg/weights/last.pt",
        "runs/detect/models/yolo/yolov8m_fish_v2/weights/best.pt" 
    ]

    model_path = None
    for path in MODEL_PATHS:
        if os.path.exists(path):
            model_path = path
            break

    if model_path is None:
        print("ERROR: No model found.")
        sys.exit(1)

    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    # ─────────────────────────────────────────────
    # Video path — change this to your aquarium video
    # ─────────────────────────────────────────────
    video_path = "datasets/fish_dataset/test/images/nini.mp4"

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

    output_path = "outputs/videos/fish_stable.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")
    print(f"Smoothing: alpha={ALPHA}, conf_alpha={CONF_ALPHA}")

    # ─────────────────────────────────────────────
    # Smoothing state per track ID
    # ─────────────────────────────────────────────
    # Stores: {track_id: {"box": [x1,y1,x2,y2], "conf": float, "last_seen": frame_num}}
    smooth_tracks = {}

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

        # ── Run tracking ──
        results = model.track(
            frame,
            imgsz=IMGSZ,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
        )

        result = results[0]
        boxes = result.boxes

        # Track which IDs we saw this frame
        seen_ids = set()

        for box in boxes:
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(float, box.xyxy[0])

            # Get track ID
            track_id = None
            if box.id is not None:
                track_id = int(box.id[0])

            if track_id is None:
                # No tracking ID — draw raw box (green)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                label = f"Fish {conf:.0%}"
                (lw, lh), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(frame, (int(x1), int(y1)-lh-bl-4), (int(x1)+lw, int(y1)), (0,255,0), -1)
                cv2.putText(frame, label, (int(x1), int(y1)-bl-2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
                continue

            seen_ids.add(track_id)
            raw_box = [x1, y1, x2, y2]

            # ── Apply EMA smoothing ──
            if track_id in smooth_tracks:
                prev = smooth_tracks[track_id]
                # Smooth bounding box coordinates
                smooth_box = [
                    ALPHA * raw_box[i] + (1 - ALPHA) * prev["box"][i]
                    for i in range(4)
                ]
                # Smooth confidence
                smooth_conf = CONF_ALPHA * conf + (1 - CONF_ALPHA) * prev["conf"]
            else:
                # First time seeing this ID — no smoothing
                smooth_box = raw_box
                smooth_conf = conf

            # Update state
            smooth_tracks[track_id] = {
                "box": smooth_box,
                "conf": smooth_conf,
                "last_seen": frame_count,
            }

            # ── Draw smoothed box ──
            sx1, sy1, sx2, sy2 = map(int, smooth_box)
            color = get_color(track_id)

            # Thicker box for stability feel
            cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), color, 2)

            # Stable label — round confidence to nearest 5% to reduce flicker
            stable_conf = round(smooth_conf * 20) / 20  # rounds to 0.05 steps
            label = f"Fish {stable_conf:.0%}"

            (lw, lh), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (sx1, sy1-lh-bl-4), (sx1+lw, sy1), color, -1)
            cv2.putText(frame, label, (sx1, sy1-bl-2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # ── Draw tracks that disappeared briefly (hold for TRACK_TIMEOUT frames) ──
        for tid, tdata in list(smooth_tracks.items()):
            if tid not in seen_ids:
                frames_since = frame_count - tdata["last_seen"]
                if frames_since <= TRACK_TIMEOUT:
                    # Keep showing the last known position (fading)
                    fade = max(0.3, 1.0 - frames_since / TRACK_TIMEOUT)
                    sx1, sy1, sx2, sy2 = map(int, tdata["box"])
                    color = get_color(tid)
                    faded_color = tuple(int(c * fade) for c in color)
                    cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), faded_color, 1)
                else:
                    # Remove stale track
                    del smooth_tracks[tid]

        # ── Info overlay ──
        fish_count = len(seen_ids)
        cv2.putText(frame, f"Fish: {fish_count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Progress
        progress = frame_count / max(total_frames, 1)
        bar_w = 200
        cv2.rectangle(frame, (20, height-30), (20+bar_w, height-15), (80,80,80), -1)
        cv2.rectangle(frame, (20, height-30), (20+int(bar_w*progress), height-15), (0,255,255), -1)
        cv2.putText(frame, f"{progress*100:.0f}%", (230, height-17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

        out.write(frame)

        cv2.imshow("Fish Detection (Stable)", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

        if frame_count % 100 == 0:
            print(f"Processed {frame_count}/{total_frames} ({progress*100:.0f}%)")

    # ─────────────────────────────────────────────
    # Cleanup
    # ─────────────────────────────────────────────
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"\nVideo saved to: {output_path}")
    print(f"Total frames processed: {frame_count}")
