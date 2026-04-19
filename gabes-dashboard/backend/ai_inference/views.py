"""
AI Inference Views — Production-grade PPE and Marine detection.

PPE detection ports the exact multi-model, multi-person pipeline:
  - helmet_model  → hat/hard-hat detection
  - mask_model    → face mask (with_mask / without_mask / incorrect_mask)
  - gasmask_model → gas mask detection
  - temporal stability voting via per-person deque history
  - person-box constructed from face/head detections

Marine detection ports the fish behavior analysis:
  - YOLO + ByteTrack for persistent fish re-identification
  - per-fish speed, acceleration, direction-change, path-straightness
  - contamination verdict with hysteresis buffer
  - per-user tracking state maintained server-side

Both endpoints return annotated_image (base64 JPEG) for live display.
"""

import base64
import io
import time
import math
import logging
import threading
from pathlib import Path
from collections import deque, Counter, defaultdict

import cv2
import numpy as np
from PIL import Image

from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from rest_framework.response import Response
from accounts.permissions import IsWorkerOrAdmin, IsFarmerOrAdmin

logger = logging.getLogger(__name__)

# ── Model directory ───────────────────────────────────────────────────────────
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# ── Model cache ───────────────────────────────────────────────────────────────
_model_cache: dict = {}
_model_lock = threading.Lock()


def load_yolo(filename: str):
    with _model_lock:
        if filename not in _model_cache:
            path = MODELS_DIR / filename
            if not path.exists():
                logger.warning(f"[AI] Model not found: {path}")
                _model_cache[filename] = None
                return None
            try:
                from ultralytics import YOLO
                _model_cache[filename] = YOLO(str(path))
                logger.info(f"[AI] Loaded: {filename}")
            except Exception as e:
                logger.error(f"[AI] Failed to load {filename}: {e}")
                _model_cache[filename] = None
        return _model_cache[filename]


# ── Image I/O helpers ─────────────────────────────────────────────────────────

def request_to_cv2(request) -> tuple:
    """Return (bgr_ndarray, error_response)."""
    if "file" in request.FILES:
        try:
            data = request.FILES["file"].read()
            arr = np.frombuffer(data, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("cv2.imdecode returned None")
            return img, None
        except Exception as e:
            return None, Response({"error": f"Invalid image file: {e}"}, status=400)

    b64 = request.data.get("frame")
    if b64:
        try:
            if "," in b64:
                b64 = b64.split(",")[1]
            data = base64.b64decode(b64)
            arr = np.frombuffer(data, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("cv2.imdecode returned None")
            return img, None
        except Exception as e:
            return None, Response({"error": f"Invalid base64 frame: {e}"}, status=400)

    return None, Response({"error": "Provide 'file' or 'frame' field."}, status=400)


def cv2_to_b64(img: np.ndarray, quality: int = 80) -> str:
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


# ─────────────────────────────────────────────────────────────────────────────
#  PPE — Multi-model, Multi-person, Temporal Voting
# ─────────────────────────────────────────────────────────────────────────────

# Per-user person_histories: { user_id : { person_id : { "helmet":deque, ... } } }
_ppe_histories: dict = defaultdict(dict)


def _center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def _distance(c1, c2):
    return math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)


def _face_to_person_box(face_box, fw, fh, frame_w, frame_h):
    """Expand a face/head box outward to cover the whole person body."""
    x1, y1, x2, y2 = face_box
    w, h = x2 - x1, y2 - y1
    px1 = max(0, x1 - int(w * 0.8))
    py1 = max(0, y1 - int(h * 1.2))
    px2 = min(frame_w, x2 + int(w * 0.8))
    py2 = min(frame_h, y2 + int(h * 2.8))
    return px1, py1, px2, py2


def _stable_vote(history, default):
    if not history:
        return default
    return Counter(history).most_common(1)[0][0]


def _get_person_id(px1, py1, px2, py2):
    cx = (px1 + px2) // 2
    cy = (py1 + py2) // 2
    return f"{cx // 50}_{cy // 50}"


class PPEAnalyzeView(APIView):
    """
    POST /api/ppe/analyze/

    Multi-model PPE detection:
      - helmet_detection.pt  → helmet
      - mask_detection.pt    → face mask (0=with, 1=without, 2=incorrect)
      - gazmask_detection.pt → gas mask

    Returns annotated frame with per-person SAFE/UNSAFE overlay.
    """
    permission_classes = [IsWorkerOrAdmin]
    parser_classes = [MultiPartParser, FormParser, JSONParser]

    def post(self, request):
        frame, err = request_to_cv2(request)
        if err:
            return err

        # Resize to working resolution
        frame = cv2.resize(frame, (640, 480))
        h, w = frame.shape[:2]

        helmet_model = load_yolo("helmet_detection.pt")
        mask_model   = load_yolo("mask_detection.pt")
        gasmask_model = load_yolo("gazmask_detection.pt")

        output = frame.copy()

        # ── Run models ────────────────────────────────────────────────────────
        helmet_hats = []       # (x1,y1,x2,y2, conf)
        mask_detections = []   # (x1,y1,x2,y2, cls_id, conf)
        gasmask_boxes = []     # (x1,y1,x2,y2, conf)

        if helmet_model:
            for box in helmet_model(frame, verbose=False)[0].boxes:
                conf = float(box.conf[0])
                if conf < 0.15:
                    continue
                cls_id = int(box.cls[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if cls_id == 0:  # hat class
                    helmet_hats.append((x1, y1, x2, y2, conf))

        if mask_model:
            for box in mask_model(frame, verbose=False)[0].boxes:
                conf = float(box.conf[0])
                if conf < 0.15:
                    continue
                cls_id = int(box.cls[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                mask_detections.append((x1, y1, x2, y2, cls_id, conf))

        if gasmask_model:
            for box in gasmask_model(frame, verbose=False)[0].boxes:
                conf = float(box.conf[0])
                # Raised from 0.25 to 0.75 to prevent false positive gas mask detections
                if conf < 0.75:
                    continue
                cls_id = int(box.cls[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if cls_id == 1:  # gasmask class
                    gasmask_boxes.append((x1, y1, x2, y2, conf))

        # ── Build person boxes from detections ────────────────────────────────
        person_candidates = []

        for m in mask_detections:
            person_candidates.append(_face_to_person_box(m[:4], m[0], m[1], w, h))

        for g in gasmask_boxes:
            person_candidates.append(_face_to_person_box(g[:4], g[0], g[1], w, h))

        for hb in helmet_hats:
            x1, y1, x2, y2, _ = hb
            hw, hh = x2 - x1, y2 - y1
            px1 = max(0, x1 - int(hw * 0.5))
            py1 = max(0, y1 - int(hh * 0.2))
            px2 = min(w, x2 + int(hw * 0.5))
            py2 = min(h, y2 + int(hh * 2.2))
            person_candidates.append((px1, py1, px2, py2))

        # Deduplicate overlapping person boxes
        final_persons = []
        for candidate in person_candidates:
            cx, cy = _center(candidate)
            if not any(_distance((cx, cy), _center(e)) < 120 for e in final_persons):
                final_persons.append(candidate)

        # If no detections at all, add a full-frame fallback so we still show something
        if not final_persons and (helmet_hats or mask_detections or gasmask_boxes):
            final_persons = [(0, 0, w, h)]

        # ── Per-person PPE matching + temporal voting ─────────────────────────
        user_id = str(request.user.id)
        hist_store = _ppe_histories[user_id]

        safe_count = 0
        unsafe_count = 0
        detections_out = []

        for person_box in final_persons:
            px1, py1, px2, py2 = person_box
            pid = _get_person_id(px1, py1, px2, py2)

            if pid not in hist_store:
                hist_store[pid] = {
                    "helmet":  deque(maxlen=10),
                    "mask":    deque(maxlen=10),
                    "gasmask": deque(maxlen=10),
                    "safe":    deque(maxlen=10),
                }
            hist = hist_store[pid]

            current_helmet  = "No Helmet"
            current_mask    = "No Mask"
            current_gasmask = "No Gas Mask"

            # Match helmet
            for (hx1, hy1, hx2, hy2, _) in helmet_hats:
                cx, cy = _center((hx1, hy1, hx2, hy2))
                if px1 <= cx <= px2 and py1 <= cy <= py1 + (py2 - py1) * 0.55:
                    current_helmet = "Helmet"
                    break

            # Match face mask (best confidence within person box)
            best_mask_conf = 0
            best_mask_label = "No Mask"
            for (mx1, my1, mx2, my2, mcls, mconf) in mask_detections:
                cx, cy = _center((mx1, my1, mx2, my2))
                if px1 <= cx <= px2 and py1 <= cy <= py1 + (py2 - py1) * 0.75:
                    if mconf > best_mask_conf:
                        best_mask_conf = mconf
                        best_mask_label = ["With Mask", "Without Mask", "Incorrect Mask"][min(mcls, 2)]
            current_mask = best_mask_label

            # Match gas mask
            for (gx1, gy1, gx2, gy2, _) in gasmask_boxes:
                cx, cy = _center((gx1, gy1, gx2, gy2))
                if px1 <= cx <= px2 and py1 <= cy <= py1 + (py2 - py1) * 0.75:
                    current_gasmask = "Gas Mask"
                    break

            # Push to temporal history
            hist["helmet"].append(current_helmet)
            hist["mask"].append(current_mask)
            hist["gasmask"].append(current_gasmask)

            helmet_status  = _stable_vote(hist["helmet"],  "No Helmet")
            mask_status    = _stable_vote(hist["mask"],    "No Mask")
            gasmask_status = _stable_vote(hist["gasmask"], "No Gas Mask")

            is_safe = (
                helmet_status == "Helmet" and
                (mask_status == "With Mask" or gasmask_status == "Gas Mask")
            )
            hist["safe"].append("SAFE" if is_safe else "UNSAFE")
            overall_status = _stable_vote(hist["safe"], "UNSAFE")

            # Draw person box
            color = (0, 220, 80) if overall_status == "SAFE" else (30, 30, 240)
            cv2.rectangle(output, (px1, py1), (px2, py2), color, 3)

            lx, ly = px1, max(30, py1 - 90)
            cv2.putText(output, overall_status, (lx, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 3)
            cv2.putText(output, helmet_status,  (lx, ly + 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
            cv2.putText(output, mask_status,    (lx, ly + 54),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
            cv2.putText(output, gasmask_status, (lx, ly + 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

            if overall_status == "SAFE":
                safe_count += 1
            else:
                unsafe_count += 1

            has_helmet  = helmet_status == "Helmet"
            has_mask    = mask_status in ("With Mask", "Gas Mask")
            has_gasmask = gasmask_status == "Gas Mask"

            detections_out.append({
                "worker_id":    len(detections_out) + 1,
                "helmet":       has_helmet,
                "gas_mask":     has_gasmask or has_mask,
                "mask_status":  mask_status,
                "gasmask_status": gasmask_status,
                "overall":      overall_status,
                "bbox":         [px1, py1, px2, py2],
            })

        # ── Global status banner ──────────────────────────────────────────────
        total = len(final_persons)
        if unsafe_count > 0:
            global_status = "UNSAFE"
            banner_color  = (30, 30, 240)
        elif safe_count > 0:
            global_status = "SAFE"
            banner_color  = (0, 200, 60)
        else:
            global_status = "NO PERSON DETECTED"
            banner_color  = (0, 140, 255)

        cv2.rectangle(output, (0, 0), (w, 44), banner_color, -1)
        cv2.putText(output, global_status, (14, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        stats = f"Persons: {total}  Safe: {safe_count}  Unsafe: {unsafe_count}"
        cv2.putText(output, stats, (14, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        return Response({
            "module":           "ppe",
            "status":           "safe" if unsafe_count == 0 and safe_count > 0 else "violation",
            "workers_detected": total,
            "violations":       unsafe_count,
            "detections":       detections_out,
            "summary":          f"{unsafe_count} violation(s) among {total} worker(s).",
            "confidence_avg":   round(safe_count / max(total, 1), 3),
            "annotated_image":  cv2_to_b64(output),
            "inference_mode":   "yolo_real" if (helmet_model or mask_model or gasmask_model) else "demo_stub",
            "processed_at":     time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })


# ─────────────────────────────────────────────────────────────────────────────
#  Marine — Fish ReID + Behavior Analysis
# ─────────────────────────────────────────────────────────────────────────────

# Configuration (from original script)
CONF_THRESHOLD  = 0.25
IOU_THRESHOLD   = 0.45
IMGSZ           = 416
ALPHA           = 0.4          # EMA smoothing for bounding box
CONF_ALPHA      = 0.3
TRACK_TIMEOUT   = 15           # frames to keep a lost track visible
REID_LOST_TIMEOUT = 300
REID_HIST_THRESH  = 0.65
REID_POS_WEIGHT   = 0.5
HISTORY_LENGTH  = 30
SPEED_THRESHOLD = 10.0
ACCEL_THRESHOLD = 4.0
DIR_THRESHOLD   = 40.0
VERDICT_BUFFER  = 45           # ~3 sec at 15fps (web inference rate)
STRESSED_RATIO_THRESH = 0.35


def _compute_histogram(frame, box):
    x1, y1, x2, y2 = map(int, box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    if x2 - x1 < 5 or y2 - y1 < 5:
        return None
    crop = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist


def _compare_fish(a, b):
    score, total_w = 0.0, 0.0
    if a["hist"] is not None and b["hist"] is not None:
        sim = cv2.compareHist(a["hist"], b["hist"], cv2.HISTCMP_CORREL)
        score += max(0.0, sim) * 4.0
        total_w += 4.0
    dx, dy = a["cx"] - b["cx"], a["cy"] - b["cy"]
    pos_sim = max(0.0, 1.0 - math.sqrt(dx**2 + dy**2) / 300.0)
    score += pos_sim * REID_POS_WEIGHT
    total_w += REID_POS_WEIGHT
    area_a = a["w"] * a["h"]
    area_b = b["w"] * b["h"]
    if max(area_a, area_b) > 0:
        size_sim = min(area_a, area_b) / max(area_a, area_b)
        score += size_sim * 2.0
        total_w += 2.0
    ar_a = a["w"] / max(1, a["h"])
    ar_b = b["w"] / max(1, b["h"])
    ar_sim = min(ar_a, ar_b) / max(ar_a, ar_b)
    score += ar_sim * 1.5
    total_w += 1.5
    return score / total_w if total_w > 0 else 0.0


def _calc_angle(p1, p2, p3):
    v1 = (p1[0] - p2[0], p1[1] - p2[1])
    v2 = (p3[0] - p2[0], p3[1] - p2[1])
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
    mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return math.degrees(math.acos(max(-1.0, min(1.0, dot / (mag1 * mag2)))))


def _fish_color(track_id, stressed):
    if stressed:
        return (30, 30, 240)  # Red (BGR)
    np.random.seed(int(track_id) * 7 + 13)
    hue = int(np.random.randint(0, 50)) + 70
    c = cv2.cvtColor(np.uint8([[[hue, 255, 200]]]), cv2.COLOR_HSV2BGR)[0][0]
    return tuple(int(x) for x in c)


# Per-user marine tracking state
_marine_states: dict = {}


def _get_marine_state(user_id: str) -> dict:
    if user_id not in _marine_states:
        _marine_states[user_id] = {
            "active_fish":    {},
            "lost_fish":      {},
            "id_remap":       {},
            "next_stable_id": 1,
            "smooth_tracks":  {},
            "behavior":       defaultdict(lambda: {
                "positions":  deque(maxlen=HISTORY_LENGTH),
                "status":     "normal",
                "speed":      0.0,
                "prev_speed": 0.0,
                "direction_change": 0.0,
            }),
            "water_verdict":   "SAFE",
            "verdict_counter": 0,
            "frame_count":     0,
        }
    return _marine_states[user_id]


class MarineAnalyzeView(APIView):
    """
    POST /api/marine/analyze/

    Fish behavior analysis with ReID and contamination detection.
    Returns annotated frame with per-fish status and water quality verdict.
    """
    permission_classes = [IsWorkerOrAdmin]
    parser_classes = [MultiPartParser, FormParser, JSONParser]

    def post(self, request):
        frame, err = request_to_cv2(request)
        if err:
            return err

        frame = cv2.resize(frame, (640, 480))
        h, w = frame.shape[:2]

        fish_model = load_yolo("fish_detection.pt")
        user_id = str(request.user.id)
        state = _get_marine_state(user_id)
        state["frame_count"] += 1
        fc = state["frame_count"]

        output = frame.copy()

        if not fish_model:
            # Fallback: return frame with a watermark
            cv2.rectangle(output, (0, 0), (w, 44), (255, 140, 0), -1)
            cv2.putText(output, "DEMO — fish_detection.pt not loaded", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            return Response({
                "module": "marine", "status": "normal",
                "fish_count": 0, "erratic_count": 0, "choking_count": 0,
                "contamination_probability": 0,
                "fish_tracks": [], "alert": "Model not loaded.",
                "annotated_image": cv2_to_b64(output),
                "inference_mode": "no_model",
                "processed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            })

        # ── Run ByteTrack ─────────────────────────────────────────────────────
        results = fish_model.track(
            frame, imgsz=IMGSZ, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD,
            persist=True, tracker="bytetrack.yaml", verbose=False,
        )
        boxes = results[0].boxes
        seen_ids   = set()
        stressed_c = 0
        erratic_c  = 0
        choking_c  = 0
        fish_tracks_out = []

        for box in boxes:
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            bt_id = int(box.id[0]) if box.id is not None else None
            if bt_id is None:
                continue

            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            hist = _compute_histogram(frame, [x1, y1, x2, y2])
            sig = {"cx": cx, "cy": cy, "w": x2-x1, "h": y2-y1, "hist": hist, "last_seen": fc}

            # ── ReID ──────────────────────────────────────────────────────────
            if bt_id in state["id_remap"]:
                stable_id = state["id_remap"][bt_id]
            else:
                best_id, best_score = None, 0.0
                for lid, lsig in state["lost_fish"].items():
                    sc = _compare_fish(sig, lsig)
                    if sc > best_score:
                        best_score, best_id = sc, lid
                if best_id and best_score >= REID_HIST_THRESH:
                    stable_id = best_id
                    del state["lost_fish"][best_id]
                else:
                    stable_id = state["next_stable_id"]
                    state["next_stable_id"] += 1
                state["id_remap"][bt_id] = stable_id

            seen_ids.add(stable_id)
            state["active_fish"][stable_id] = {**sig}

            # ── Behavior ─────────────────────────────────────────────────────
            bh = state["behavior"][stable_id]
            bh["positions"].append((cx, cy))
            pts = list(bh["positions"])

            speed, dir_change, accel = 0.0, 0.0, 0.0
            if len(pts) >= 5:
                dx_ = pts[-1][0] - pts[-5][0]
                dy_ = pts[-1][1] - pts[-5][1]
                speed = math.sqrt(dx_**2 + dy_**2) / 5.0
            if len(pts) >= 10:
                dir_change = _calc_angle(pts[-10], pts[-5], pts[-1])
            accel = speed - bh.get("prev_speed", 0.0)
            bh["prev_speed"] = speed

            # Path straightness
            straightness, l_path, d_straight = 1.0, 0.0, 0.0
            if len(pts) >= 15:
                d_straight = math.sqrt((pts[-1][0]-pts[0][0])**2 + (pts[-1][1]-pts[0][1])**2)
                l_path = sum(
                    math.sqrt((pts[i][0]-pts[i-1][0])**2 + (pts[i][1]-pts[i-1][1])**2)
                    for i in range(1, len(pts))
                )
                if l_path > 5.0:
                    straightness = d_straight / l_path

            is_darting   = accel > ACCEL_THRESHOLD and speed > 5.0
            is_random    = straightness < 0.65 and l_path > 15.0 and d_straight > 25.0
            is_stressed  = (
                speed > SPEED_THRESHOLD or
                (dir_change > DIR_THRESHOLD and speed > 4.0) or
                is_random or is_darting
            )

            if is_stressed:
                bh["status"] = "stressed"
            elif speed < SPEED_THRESHOLD * 0.7 and not is_random:
                bh["status"] = "normal"

            if bh["status"] == "stressed":
                stressed_c += 1

            # Map to behavior labels for UI
            behavior_label = "normal"
            if bh["status"] == "stressed":
                if accel > ACCEL_THRESHOLD:
                    behavior_label = "erratic"
                    erratic_c += 1
                else:
                    behavior_label = "choking"
                    choking_c += 1

            fish_tracks_out.append({
                "fish_id":             stable_id,
                "behavior":            behavior_label,
                "velocity":            round(speed, 2),
                "direction_change_rate": round(dir_change, 2),
            })

            # ── Smooth box + draw ─────────────────────────────────────────────
            raw = [x1, y1, x2, y2]
            if stable_id in state["smooth_tracks"]:
                prev_b = state["smooth_tracks"][stable_id]["box"]
                smooth = [ALPHA * raw[i] + (1 - ALPHA) * prev_b[i] for i in range(4)]
                smooth_conf = CONF_ALPHA * conf + (1-CONF_ALPHA) * state["smooth_tracks"][stable_id]["conf"]
            else:
                smooth, smooth_conf = raw, conf
            state["smooth_tracks"][stable_id] = {"box": smooth, "conf": smooth_conf, "last_seen": fc}

            sx1, sy1, sx2, sy2 = map(int, smooth)
            color = _fish_color(stable_id, bh["status"] == "stressed")

            # trajectory line
            for i in range(1, len(pts)):
                thickness = max(1, int(2 * i / len(pts)))
                cv2.line(output, (int(pts[i-1][0]), int(pts[i-1][1])),
                                 (int(pts[i][0]),   int(pts[i][1])), color, thickness)

            cv2.rectangle(output, (sx1, sy1), (sx2, sy2), color, 2)
            label = f"Fish #{stable_id}  {smooth_conf:.0%}  spd:{speed:.1f}"
            (lw_, lh_), bl_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 2)
            cv2.rectangle(output, (sx1, sy1 - lh_ - bl_ - 4), (sx1 + lw_, sy1), color, -1)
            cv2.putText(output, label, (sx1, sy1 - bl_ - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 2)

        # ── Cleanup lost fish ─────────────────────────────────────────────────
        for sid in list(state["active_fish"].keys()):
            if sid not in seen_ids:
                info = state["active_fish"][sid]
                if fc - info["last_seen"] > TRACK_TIMEOUT:
                    state["lost_fish"][sid] = info
                    del state["active_fish"][sid]
                    for bt, st in list(state["id_remap"].items()):
                        if st == sid:
                            del state["id_remap"][bt]

        for lid in list(state["lost_fish"].keys()):
            if fc - state["lost_fish"][lid]["last_seen"] > REID_LOST_TIMEOUT:
                del state["lost_fish"][lid]

        # ── Water verdict with hysteresis ─────────────────────────────────────
        active_count = len(seen_ids)
        if active_count > 0:
            stressed_ratio = stressed_c / active_count
            is_contaminated_now = stressed_ratio >= STRESSED_RATIO_THRESH

            if is_contaminated_now and state["water_verdict"] == "SAFE":
                state["verdict_counter"] += 1
                if state["verdict_counter"] >= VERDICT_BUFFER:
                    state["water_verdict"] = "CONTAMINATED"
                    state["verdict_counter"] = 0
            elif not is_contaminated_now and state["water_verdict"] == "CONTAMINATED":
                state["verdict_counter"] += 1
                if state["verdict_counter"] >= VERDICT_BUFFER:
                    state["water_verdict"] = "SAFE"
                    state["verdict_counter"] = 0
            else:
                state["verdict_counter"] = max(0, state["verdict_counter"] - 1)

        verdict = state["water_verdict"]

        # ── Verdict banner overlay ────────────────────────────────────────────
        banner_color = (30, 30, 240) if verdict == "CONTAMINATED" else (0, 200, 60)
        cv2.rectangle(output, (0, 0), (w, 48), banner_color, -1)
        cv2.putText(output, f"WATER QUALITY: {verdict}", (14, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2)
        cv2.putText(output, f"Stressed: {stressed_c}/{active_count}", (14, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 240, 240), 2)

        contamination_prob = round(stressed_c / max(active_count, 1), 3) if active_count else 0.0

        return Response({
            "module":                    "marine",
            "status":                    "contaminated" if verdict == "CONTAMINATED" else "normal",
            "fish_count":                active_count,
            "erratic_count":             erratic_c,
            "choking_count":             choking_c,
            "contamination_probability": contamination_prob,
            "fish_tracks":               fish_tracks_out,
            "alert": ("Phosphogypsum contamination likely — alert marine team!"
                      if verdict == "CONTAMINATED" else "Water quality appears normal."),
            "annotated_image":  cv2_to_b64(output),
            "inference_mode":   "yolo_real",
            "processed_at":     time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })


# ─────────────────────────────────────────────────────────────────────────────
#  Land — Satellite Segmentation (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

import random
from PIL import Image as PILImage, ImageDraw


def _pil_to_b64(img, quality=80):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _draw_box_pil(draw, box, label, conf, color, w, h):
    x1, y1, x2, y2 = map(int, box)
    for d in range(3):
        draw.rectangle([x1 - d, y1 - d, x2 + d, y2 + d], outline=color)
    txt = f"{label} {conf:.0%}"
    cw, ch = 7, 13
    tw = len(txt) * cw + 8
    th = ch + 6
    ly1 = max(0, y1 - th - 2)
    draw.rectangle([x1, ly1, x1 + tw, ly1 + th], fill=color)
    draw.text((x1 + 4, ly1 + 3), txt, fill=(10, 10, 10))



# ─────────────────────────────────────────────────────────────────────────────
#  Land — UNet Segmentation + CSV Crop Recommendation
# ─────────────────────────────────────────────────────────────────────────────

import csv
import torch
import torchvision.transforms.functional as TVF

# ── Land class definitions (matching training labels) ─────────────────────────
LAND_CLASSES = {
    0: {"name": "urban_land",       "color": (0, 255, 255),   "label": "Urban / Industrial"},
    1: {"name": "agriculture_land", "color": (255, 255, 0),   "label": "Agriculture Land"},
    2: {"name": "rangeland",        "color": (255, 0, 255),   "label": "Rangeland"},
    3: {"name": "forest_land",      "color": (0, 255, 0),     "label": "Forest / Vegetation"},
    4: {"name": "water",            "color": (0, 0, 255),     "label": "Water Body"},
    5: {"name": "barren_land",      "color": (255, 255, 255), "label": "Barren / Saline Land"},
    6: {"name": "unknown",          "color": (80, 80, 80),    "label": "Unknown"},
}
LAND_IMAGE_SIZE = (512, 512)
NUM_LAND_CLASSES = 7

# Dominant land class → detected land type for CSV lookup
CLASS_TO_LANDTYPE = {
    "urban_land":       "urban",
    "agriculture_land": "agriculture",
    "rangeland":        "rangeland",
    "forest_land":      "forest",
    "water":            "water",
    "barren_land":      "barren",
    "unknown":          "unknown",
}

# Suitability of each land type for agriculture
LAND_SUITABILITY = {
    "agriculture": "high",
    "rangeland":   "medium",
    "forest":      "medium",
    "barren":      "low",
    "water":       "unsuitable",
    "urban":       "unsuitable",
    "unknown":     "unknown",
}


def _load_land_model():
    """Load the UNet model with ResNet34 encoder (lazy + cached)."""
    if "land_unet" in _model_cache:
        return _model_cache["land_unet"]
    with _model_lock:
        if "land_unet" in _model_cache:
            return _model_cache["land_unet"]
        # Prefer unet_finetuned.pth exactly like the custom script 
        path = MODELS_DIR / "unet_finetuned.pth"
        if not path.exists():
            path = MODELS_DIR / "land_regeneration.pth"
            
        if not path.exists():
            logger.warning("[Land] Model weights not found")
            _model_cache["land_unet"] = None
            return None
        try:
            import segmentation_models_pytorch as smp
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = smp.Unet(
                encoder_name="resnet34",
                encoder_weights=None,
                in_channels=3,
                classes=NUM_LAND_CLASSES,
            )
            model.load_state_dict(torch.load(str(path), map_location=device))
            model.to(device)
            model.eval()
            _model_cache["land_unet"] = model
            _model_cache["land_device"] = device
            logger.info("[Land] UNet model loaded successfully")
            return model
        except Exception as e:
            logger.error(f"[Land] Failed to load UNet: {e}")
            _model_cache["land_unet"] = None
            return None


def _mask_to_rgb(mask_idx: np.ndarray) -> np.ndarray:
    """Convert class-index mask to RGB image."""
    h, w = mask_idx.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for idx, cfg in LAND_CLASSES.items():
        r, g, b = cfg["color"]
        rgb[mask_idx == idx] = [r, g, b]
    return rgb


def _load_crop_csv() -> list:
    """Load the CSV once and return rows as list of dicts."""
    if "crop_csv" in _model_cache:
        return _model_cache["crop_csv"]
    path = MODELS_DIR / "gabes_crop_mapping.csv"
    if not path.exists():
        _model_cache["crop_csv"] = []
        return []
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    _model_cache["crop_csv"] = rows
    return rows


def _get_crop_recommendations(zone: str, num: int = 5) -> list:
    """Return top N high-suitability crops for the given zone."""
    rows = _load_crop_csv()
    matches = [r for r in rows if r["zone"].lower() == zone.lower()
               and r.get("suitability", "").lower() == "high"]
    if not matches:
        matches = [r for r in rows if r["zone"].lower() == zone.lower()]
    # deduplicate by crop name
    seen, result = set(), []
    for r in matches:
        if r["crop"] not in seen:
            seen.add(r["crop"])
            result.append(r)
        if len(result) >= num:
            break
    return result


def _generate_explanation(zone: str, detected_land: str, dominant_class: str,
                           soil_type: str, salinity: str, top_crop: dict,
                           class_breakdown: dict, land_size_m2: float) -> str:
    """
    Generate a structured AI explanation following the prompt template.
    Output mirrors the exact format the user specified.
    """
    suitability = LAND_SUITABILITY.get(detected_land, "unknown")
    crop_name = top_crop.get("crop", "—") if top_crop else "—"
    water_need = float(top_crop.get("water_need_l_per_m2_day", 0)) if top_crop else 0
    total_water = round(water_need * land_size_m2, 1) if land_size_m2 > 0 else None

    zone_display = zone.replace("_", " ").title()
    dominant_label = LAND_CLASSES.get(
        next((k for k, v in LAND_CLASSES.items() if v["name"] == dominant_class), 0), {}
    ).get("label", dominant_class)

    # Summary sentence
    if suitability == "high":
        summary = (f"The satellite scan of {zone_display} reveals predominantly "
                   f"{dominant_label.lower()} — excellent conditions for sustainable farming.")
    elif suitability == "medium":
        summary = (f"The scan shows {dominant_label.lower()} in {zone_display}. "
                   f"Farming is possible with some preparation.")
    elif suitability == "low":
        summary = (f"The scan detects significant barren or degraded land in {zone_display}. "
                   f"Rehabilitation is needed before cultivation.")
    else:
        summary = (f"The scan of {zone_display} reveals {dominant_label.lower()} — "
                   f"this area is not suitable for agriculture at this time.")

    # Build breakdown line
    top_classes = sorted(class_breakdown.items(), key=lambda x: -x[1])[:3]
    breakdown_str = " | ".join(
        f"{LAND_CLASSES.get(k, {}).get('label', k)}: {v:.1f}%"
        for k, v in top_classes if v > 0.5
    )

    # Soil description
    soil_map = {
        "alluvial_fertile": "Rich alluvial soil — high water retention, excellent for crops",
        "arid_sandy":       "Sandy, arid soil — low water retention, drought-resistant crops preferred",
        "halomorphic":      "Salt-affected halomorphic soil — only salt-tolerant crops recommended",
        "rocky":            "Rocky terrain — limited depth, terrace farming possible",
        "mixed":            "Mixed soil composition — moderate fertility",
    }
    soil_desc = soil_map.get(soil_type, f"{soil_type} soil")
    salinity_desc = {
        "low":      "Salinity is within normal range — no restrictions",
        "medium":   "Moderate salinity — choose salt-tolerant varieties",
        "high":     "High salinity — only halophyte crops advisable",
        "very_high":"Very high salinity — land requires desalination treatment first",
    }.get(salinity, f"Salinity level: {salinity}")

    # Crop reason
    crop_reasons = {
        "olive":       "Extremely drought-tolerant, thrives in Mediterranean and arid soils",
        "date palm":   "Native to arid Saharan zones, highly productive with minimal water in Gabes",
        "cactus":      "Requires almost no water — ideal for very dry or previously barren land",
        "moringa":     "Fast-growing, high-value, tolerates poor soil and drought conditions",
        "barley":      "One of the most salt-tolerant cereal crops, low water demand",
        "fig":         "Deep-rooted, drought-resistant, adapts well to Gabes climate",
        "pistachio":   "Heat and drought tolerant, high commercial value in this region",
        "pomegranate": "Excellent salt tolerance, low water needs, suited to the Gabes oasis",
        "aloe vera":   "Succulent with minimal water needs — supports land rehabilitation",
        "sesame":      "Drought-resistant, short growing season, good for sandy soils",
    }
    crop_reason = crop_reasons.get(crop_name.lower(),
                                   "Well-suited to this zone's climate and soil conditions")

    # Water note
    water_line = f"{water_need} L/m²/day"
    if total_water:
        water_line += f" · Total for your {int(land_size_m2):,} m² plot: ~{total_water:,.0f} L/day"

    # Warning
    warning = ""
    if suitability in ("low", "unsuitable"):
        warning = (
            "\n⚠️ Advice: This land type requires treatment before farming. "
            "Consider soil remediation, halophyte cover crops, or consulting a local agronomist "
            "before any investment."
        )
    elif salinity in ("high", "very_high"):
        warning = (
            "\n⚠️ Advice: High salinity detected. Use drip irrigation to avoid salt accumulation "
            "and select certified salt-tolerant seed varieties only."
        )

    lines = [
        summary,
        "",
        f"📍 Location: {zone_display}",
        f"🌍 Land condition: {dominant_label} ({breakdown_str})",
        f"🌱 Soil analysis: {soil_desc}. {salinity_desc}.",
        f"🌳 Recommended crop: {crop_name.title()} — {crop_reason}.",
        f"💧 Water requirement: {water_line}.",
    ]
    if warning:
        lines.append(warning)

    return "\n".join(lines)


class LandAnalyzeView(APIView):
    """
    POST /api/land/analyze/

    Full pipeline:
      1. UNet + ResNet34 segmentation on the uploaded/streamed image
      2. Pixel-level class breakdown (7 land types)
      3. CSV crop recommendation for the selected Gabes zone
      4. Structured AI explanation (farmer-friendly, follows the prompt template)

    Access: approved farmers and admins only.
    Workers / technicians cannot access land analysis.

    Form fields:
      - file / frame : image
      - zone         : Gabes zone name (e.g. Gabes_oasis)
      - land_size_m2 : optional float (plot size for water calculation)
    """
    permission_classes = [IsFarmerOrAdmin]
    parser_classes = [MultiPartParser, FormParser, JSONParser]

    def post(self, request):
        # ── Parse image ──────────────────────────────────────────────────────
        if "file" in request.FILES:
            try:
                data = request.FILES["file"].read()
                arr = np.frombuffer(data, np.uint8)
                bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if bgr is None: raise ValueError("Could not decode image")
            except Exception as e:
                return Response({"error": str(e)}, status=400)
        else:
            b64 = request.data.get("frame", "")
            if not b64:
                return Response({"error": "Provide 'file' or 'frame'."}, status=400)
            try:
                if "," in b64: b64 = b64.split(",")[1]
                arr = np.frombuffer(base64.b64decode(b64), np.uint8)
                bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if bgr is None: raise ValueError("Could not decode base64 image")
            except Exception as e:
                return Response({"error": str(e)}, status=400)

        # ── Parse form params ────────────────────────────────────────────────
        zone = request.data.get("zone", "Gabes_oasis")
        try:
            land_size_m2 = float(request.data.get("land_size_m2", 0))
        except (ValueError, TypeError):
            land_size_m2 = 0.0

        # ── Load model and run segmentation ──────────────────────────────────
        model = _load_land_model()
        device = _model_cache.get("land_device", torch.device("cpu"))

        orig_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(orig_rgb, LAND_IMAGE_SIZE)

        pred_mask  = None
        inference_mode = "demo_stub"

        if model is not None:
            try:
                # Normalize exactly as during training (ImageNet stats)
                tensor = TVF.to_tensor(resized)
                tensor = TVF.normalize(tensor,
                                       mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
                inp = tensor.unsqueeze(0).to(device)

                with torch.no_grad():
                    out = model(inp)
                    
                    # Boost agriculture confidence (class 1) moderately 
                    out[:, 1, :, :] += 2.0
                    
                    # Boost forest confidence (class 3) stronger
                    out[:, 3, :, :] += 4.0
                    
                    # Lower water body confidence (class 4) but dialled back slightly more
                    out[:, 4, :, :] -= 1.5
                    
                    # Add a stronger boost for barren land (class 5)
                    out[:, 5, :, :] += 1.5
                    
                    pred_mask = torch.argmax(out, dim=1).squeeze(0).cpu().numpy()

                inference_mode = "unet_real"
            except Exception as e:
                logger.error(f"[Land] UNet inference failed: {e}")
                pred_mask = None

        if pred_mask is None:
            # Fallback: keep exactly as original before tweaks
            h, w = LAND_IMAGE_SIZE
            pred_mask = np.zeros((h, w), dtype=np.int64)
            gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
            g_ch = resized[:, :, 1]
            
            pred_mask[g_ch > 120] = 1   # agriculture
            pred_mask[(gray < 60)] = 5  # barren
            inference_mode = "demo_stub"

        # ── Class breakdown ──────────────────────────────────────────────────
        total_pixels = pred_mask.size
        class_breakdown = {}
        for cls_id in range(NUM_LAND_CLASSES):
            pct = float((pred_mask == cls_id).sum()) / total_pixels * 100
            class_breakdown[cls_id] = round(pct, 2)

        # Dominant class (excluding "unknown")
        dominant_cls = max(
            (k for k in class_breakdown if k != 6),
            key=lambda k: class_breakdown[k]
        )
        dominant_name = LAND_CLASSES[dominant_cls]["name"]
        dominant_label = LAND_CLASSES[dominant_cls]["label"]
        detected_land  = CLASS_TO_LANDTYPE.get(dominant_name, "unknown")

        # ── Build predicted mask RGB & overlay ───────────────────────────────
        pred_rgb = _mask_to_rgb(pred_mask)                    # RGB colour mask
        # Overlay: 60% original + 40% mask (matches original script)
        overlay  = cv2.addWeighted(resized, 0.6,
                                   cv2.cvtColor(pred_rgb, cv2.COLOR_RGB2BGR), 0.4, 0)
        overlay  = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

        # Add legend strip at bottom of overlay
        legend_h = 28
        legend = np.zeros((legend_h, LAND_IMAGE_SIZE[0], 3), dtype=np.uint8)
        top_n = sorted(class_breakdown.items(), key=lambda x: -x[1])[:4]
        block_w = LAND_IMAGE_SIZE[0] // max(len(top_n), 1)
        for i, (cls_id, pct) in enumerate(top_n):
            r, g, b = LAND_CLASSES[cls_id]["color"]
            legend[0:, i*block_w:(i+1)*block_w] = [r, g, b]
        overlay_with_legend = np.vstack([overlay, legend])

        # ── CSV crop lookup ──────────────────────────────────────────────────
        zone_rows = _get_crop_recommendations(zone, num=5)

        # Get soil / salinity from first matching row
        soil_type = zone_rows[0]["soil_type"]  if zone_rows else "unknown"
        salinity  = zone_rows[0]["salinity"]   if zone_rows else "unknown"
        top_crop  = zone_rows[0]               if zone_rows else {}

        # Serialisable crop list
        crop_list = [
            {
                "crop":       r["crop"],
                "water_need": float(r["water_need_l_per_m2_day"]),
                "suitability": r.get("suitability", "high"),
            }
            for r in zone_rows
        ]

        # ── AI explanation ───────────────────────────────────────────────────
        explanation = _generate_explanation(
            zone=zone,
            detected_land=detected_land,
            dominant_class=dominant_name,
            soil_type=soil_type,
            salinity=salinity,
            top_crop=top_crop,
            class_breakdown=class_breakdown,
            land_size_m2=land_size_m2,
        )

        # ── Encode all three views as base64 ─────────────────────────────────
        def _enc(arr_rgb):
            bgr_out = cv2.cvtColor(arr_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
            _, buf = cv2.imencode(".jpg", bgr_out, [cv2.IMWRITE_JPEG_QUALITY, 82])
            return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()

        return Response({
            "module":           "land",
            "status":           "analyzed",
            "detected_land":    detected_land,
            "dominant_class":   dominant_label,
            "class_breakdown":  {
                LAND_CLASSES[k]["label"]: v
                for k, v in class_breakdown.items() if v > 0.1
            },
            "zone":             zone,
            "soil_type":        soil_type,
            "salinity":         salinity,
            "crop_recommendations": crop_list,
            "top_crop":         top_crop.get("crop", "—"),
            "water_need_per_m2": float(top_crop.get("water_need_l_per_m2_day", 0)) if top_crop else 0,
            "total_water_liters": round(float(top_crop.get("water_need_l_per_m2_day", 0)) * land_size_m2, 1) if land_size_m2 and top_crop else None,
            "land_size_m2":     land_size_m2,
            "explanation":      explanation,
            # Three image panels (matches original script 1-2-3 layout)
            "original_image":   _enc(resized),
            "mask_image":       _enc(pred_rgb),
            "annotated_image":  _enc(overlay_with_legend),
            "inference_mode":   inference_mode,
            "processed_at":     time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })


