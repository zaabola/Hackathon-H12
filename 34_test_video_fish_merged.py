from ultralytics import YOLO
import cv2
import os

# -----------------------------
# Load the NEW merged model
# -----------------------------
model = YOLO("runs/detect/runs/detect/models/yolo/yolov8_fish_v34/weights/best.pt")

# -----------------------------
# Video path
# -----------------------------
video_path = "datasets/fish_dataset/test/images/11.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise FileNotFoundError(f"Could not open video: {video_path}")

# -----------------------------
# Output setup
# -----------------------------
os.makedirs("outputs/videos", exist_ok=True)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

output_path = "outputs/videos/fish_output_merged.mp4"

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# -----------------------------
# Process video
# -----------------------------
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Run YOLO with tuned settings
    results = model(frame, imgsz=640, conf=0.35, iou=0.45, verbose=False)
    result = results[0]
    boxes = result.boxes

    fishes = []

    for box in boxes:
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        fishes.append((x1, y1, x2, y2, conf))

    # Draw detections
    for (x1, y1, x2, y2, conf) in fishes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Fish {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display fish count
    cv2.putText(frame, f"Fish Count: {len(fishes)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Write frame
    out.write(frame)

    # Show frame
    cv2.imshow("Fish Detection Video", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

    if frame_count % 100 == 0:
        print(f"Processed {frame_count} frames...")

# -----------------------------
# Cleanup
# -----------------------------
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Video saved to: {output_path}")
print(f"Total frames processed: {frame_count}")
