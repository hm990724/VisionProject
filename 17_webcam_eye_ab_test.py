"""17_webcam_eye_ab_test.py - Same-frame YOLO26n eye baseline vs CLAHE A/B.

Uses common.to_square and roi_utils.crop_eyes from VisionProject.
Does NOT alter model weights, other scripts, training data, or camera pixels for inference.

Usage:
    python 17_webcam_eye_ab_test.py
    python 17_webcam_eye_ab_test.py --camera 1 --stride 3

Keys: Q/ESC quit, O label OPEN, C label CLOSED, U clear label,
      R reset active warning timers, S save annotated dashboard screenshot.

Manual label stays active until O/C/U is pressed again. A frame label is NOT
an independent observation; use separate recording sessions/events for evaluation.

WARNING: Research demo only; never use while actually driving.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
from ultralytics import YOLO

from common import to_square
from roi_utils import crop_eyes

ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = ROOT / "runs/eyes_yolo26n-cls/weights/best.pt"
CSV_COLUMNS = [
    "wall_time", "session_sec", "frame_number", "sample_number", "fps_display", "inference_ms",
    "face_roi_ok", "eye_roi_count", "baseline_closed_probability", "clahe_closed_probability",
    "baseline_threshold", "clahe_threshold", "baseline_prediction", "clahe_prediction",
    "baseline_closed_duration_sec", "clahe_closed_duration_sec", "baseline_warning_event",
    "clahe_warning_event", "baseline_warning_active", "clahe_warning_active", "manual_label"
]


def clahe_eye(crop: Image.Image) -> Image.Image:
    """Exactly the LAB L-channel CLAHE recipe in 15_roi_ablation.py."""
    rgb = np.asarray(crop.convert("RGB"))
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(lab[:, :, 0])
    return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))


def predict_closed(model, images, class_index):
    """Mean of per-eye Closed probabilities; identical aggregation as 15."""
    probs = []
    for eye in images:
        prediction = model.predict(to_square(eye, 224), imgsz=224, device="cpu", verbose=False)[0]
        probs.append(float(prediction.probs.data[class_index]))
    return float(np.mean(probs)) if probs else None


class WarningTracker:
    """Hold-based warning. One event per continuous confirmed Closed episode."""
    def __init__(self, threshold, hold_sec, max_gap_sec):
        self.threshold = threshold
        self.hold_sec = hold_sec
        self.max_gap_sec = max_gap_sec
        self.since = None
        self.last_sample = None
        self.latched = False

    def reset(self):
        self.since = None
        self.last_sample = None
        self.latched = False

    def update(self, prob, timestamp):
        if self.last_sample is not None and timestamp - self.last_sample > self.max_gap_sec:
            self.reset()
        self.last_sample = timestamp
        if prob is None or prob < self.threshold:
            self.since = None
            self.latched = False
            return 0.0, False, False
        if self.since is None:
            self.since = timestamp
        duration = max(0.0, timestamp - self.since)
        event = duration >= self.hold_sec and not self.latched
        if event:
            self.latched = True
        return duration, event, duration >= self.hold_sec


def text(img, line, x, y, scale=0.56, color=(224, 238, 248), thick=1):
    cv2.putText(img, line, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thick, cv2.LINE_AA)


def panel_preview(panel, crop, x, y, title, width=120, height=80):
    text(panel, title, x, y - 7, 0.41, (144, 194, 205), 1)
    cv2.rectangle(panel, (x, y), (x + width, y + height), (40, 58, 73), -1)
    if crop is None:
        text(panel, "NO ROI", x + 16, y + height // 2, 0.45)
        return
    sample = np.asarray(to_square(crop, min(width, height)))
    sample = cv2.cvtColor(sample, cv2.COLOR_RGB2BGR)
    sample = cv2.resize(sample, (width, height), interpolation=cv2.INTER_AREA)
    panel[y:y + height, x:x + width] = sample
    cv2.rectangle(panel, (x, y), (x + width - 1, y + height - 1), (94, 129, 146), 1)


def draw_dashboard(frame, baseline, improved, trackers, extras, preview):
    """Display-only composition. The original frame remains untouched by inference."""
    h, w = frame.shape[:2]
    display_width, display_height = 720, 480
    view = cv2.resize(frame, (display_width, display_height), interpolation=cv2.INTER_AREA)
    right_width = 360
    canvas = np.full((display_height, display_width + right_width, 3), (21, 31, 43), dtype=np.uint8)
    canvas[:, :display_width] = view
    x = display_width + 18
    text(canvas, "AURA  /  EYE A-B TEST", x, 35, 0.63, (145, 243, 207), 2)
    text(canvas, "BASELINE  :  YOLO26n", x, 80, 0.51)
    for title, prob, tracker, duration, active, pos in (
        ("BASELINE", baseline, trackers[0], extras["base_duration"], extras["base_active"], 111),
        ("CLAHE", improved, trackers[1], extras["clahe_duration"], extras["clahe_active"], 229),
    ):
        if title == "CLAHE":
            text(canvas, "CLAHE (LAB, 2.0, 4x4)", x, 200, 0.51)
        value = "UNKNOWN" if prob is None else f"{prob * 100:5.1f}%"
        status = "WARN" if active else ("UNKNOWN" if prob is None else ("CLOSED" if prob >= tracker.threshold else "OPEN"))
        color = (78, 103, 240) if active else ((124, 226, 175) if status == "OPEN" else (172, 198, 213))
        text(canvas, f"P(CLOSED): {value}", x, pos + 21, 0.68, color, 2)
        text(canvas, f"thr={tracker.threshold:.2f}   {status}  {duration:.1f}s", x, pos + 47, 0.47, color)
    text(canvas, f"ROI eyes: {extras['roi_count']}   FPS: {extras['fps']:.1f}", x, 295, 0.47)
    text(canvas, f"Inference: {extras['infer_ms']:.0f} ms / sample", x, 315, 0.43)
    panel_preview(canvas, preview[0], x, 353, "ORIGINAL EYE", width=150, height=85)
    panel_preview(canvas, preview[1], x + 172, 353, "CLAHE EYE", width=150, height=85)
    if extras["manual_label"]:
        text(canvas, "MANUAL GT: " + extras["manual_label"], 18, 33, 0.7, (116, 255, 224), 2)
    if extras["base_event"] or extras["clahe_event"]:
        text(canvas, "WARNING EVENT -> LOGGED", 18, 68, 0.64, (80, 80, 245), 2)
    # Controls shown on video only; inference uses untouched frame.
    text(canvas, "Q:quit  O:open  C:closed  U:clear label  R:reset  S:screenshot", 17, 462, 0.46, (255, 255, 255))
    return canvas


def parse_args():
    p = argparse.ArgumentParser(description="Eye webcam A/B test (baseline YOLO vs LAB CLAHE)")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--stride", type=int, default=2, help="Run model on every Nth frame")
    p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    p.add_argument("--baseline-threshold", type=float, default=0.50)
    p.add_argument("--clahe-threshold", type=float, default=0.25)
    p.add_argument("--closed-seconds", type=float, default=2.0)
    p.add_argument("--max-sample-gap", type=float, default=1.5, help="Gap after which continuous duration resets")
    p.add_argument("--out", type=Path, default=ROOT / "reports/webcam_eye_ab")
    args = p.parse_args()
    if args.stride < 1 or args.width <= 0 or args.height <= 0:
        p.error("--stride, --width, --height must be positive")
    if not 0 <= args.baseline_threshold <= 1 or not 0 <= args.clahe_threshold <= 1:
        p.error("thresholds must be 0~1")
    if args.closed_seconds <= 0 or args.max_sample_gap <= 0:
        p.error("duration settings must be > 0")
    return args


def main():
    args = parse_args()
    if not args.weights.is_file():
        raise SystemExit(f"Eye model not found: {args.weights}")
    args.out.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = args.out / f"eye_ab_{run_id}.csv"
    model = YOLO(str(args.weights))
    closed_class = next((int(k) for k, label in model.names.items() if str(label).lower() == "closed"), None)
    if closed_class is None:
        raise SystemExit(f"No Closed class in YOLO labels: {model.names}")
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    cap = cv2.VideoCapture(args.camera, backend)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Cannot open camera. Try --camera 1 and close other webcam programs.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    baseline_tracker = WarningTracker(args.baseline_threshold, args.closed_seconds, args.max_sample_gap)
    clahe_tracker = WarningTracker(args.clahe_threshold, args.closed_seconds, args.max_sample_gap)
    baseline = clahe = None
    base_duration = clahe_duration = 0.0
    base_active = clahe_active = False
    base_event = clahe_event = False
    manual_label = ""
    preview = (None, None)
    infer_ms = 0.0
    fps = 0.0
    frame_number = sample_number = 0
    started = time.monotonic()
    prev_frame = started
    roi_count = 0
    print(f"[READY] Baseline thr={args.baseline_threshold}, CLAHE thr={args.clahe_threshold}")
    print("[KEYS] O=Open, C=Closed, U=Unknown label, R=Reset timers, S=screenshot, Q=Quit")
    print("[CSV]", csv_path)
    try:
        with csv_path.open("w", newline="", encoding="utf-8-sig") as out_file:
            writer = csv.DictWriter(out_file, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            with mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False, max_num_faces=1, refine_landmarks=True,
                min_detection_confidence=0.5, min_tracking_confidence=0.5,
            ) as mesh:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        print("[WARN] Camera frame unavailable, stopping.")
                        break
                    now = time.monotonic()
                    fps = 1.0 / max(1e-3, now - prev_frame)
                    prev_frame = now
                    frame_number += 1
                    base_event = clahe_event = False
                    if frame_number % args.stride == 0:
                        start_inference = time.perf_counter()
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        crops = crop_eyes(Image.fromarray(rgb), mesh)
                        roi_count = len(crops)
                        if crops:
                            enhanced = [clahe_eye(eye) for eye in crops]
                            baseline = predict_closed(model, crops, closed_class)
                            clahe = predict_closed(model, enhanced, closed_class)
                            preview = (crops[0], enhanced[0])
                        else:
                            baseline = clahe = None
                            preview = (None, None)
                        sample_time = time.monotonic()
                        base_duration, base_event, base_active = baseline_tracker.update(baseline, sample_time)
                        clahe_duration, clahe_event, clahe_active = clahe_tracker.update(clahe, sample_time)
                        infer_ms = (time.perf_counter() - start_inference) * 1000.0
                        sample_number += 1
                        row = {
                            "wall_time": datetime.now().isoformat(timespec="milliseconds"),
                            "session_sec": round(sample_time - started, 3),
                            "frame_number": frame_number, "sample_number": sample_number,
                            "fps_display": round(fps, 2), "inference_ms": round(infer_ms, 2),
                            "face_roi_ok": int(bool(crops)), "eye_roi_count": roi_count,
                            "baseline_closed_probability": "" if baseline is None else round(baseline, 6),
                            "clahe_closed_probability": "" if clahe is None else round(clahe, 6),
                            "baseline_threshold": args.baseline_threshold, "clahe_threshold": args.clahe_threshold,
                            "baseline_prediction": "UNKNOWN" if baseline is None else ("Closed" if baseline >= args.baseline_threshold else "Open"),
                            "clahe_prediction": "UNKNOWN" if clahe is None else ("Closed" if clahe >= args.clahe_threshold else "Open"),
                            "baseline_closed_duration_sec": round(base_duration, 3),
                            "clahe_closed_duration_sec": round(clahe_duration, 3),
                            "baseline_warning_event": int(base_event), "clahe_warning_event": int(clahe_event),
                            "baseline_warning_active": int(base_active), "clahe_warning_active": int(clahe_active),
                            "manual_label": manual_label,
                        }
                        writer.writerow(row)
                        out_file.flush()  # keep logs even on abrupt stop
                        if base_event or clahe_event:
                            print(f"[EVENT {sample_time - started:.1f}s] baseline={base_event} clahe={clahe_event} label={manual_label or 'UNLABELED'}")
                    rendered = draw_dashboard(frame, baseline, clahe, (baseline_tracker, clahe_tracker),
                        {"base_duration": base_duration, "clahe_duration": clahe_duration,
                         "base_active": base_active, "clahe_active": clahe_active,
                         "base_event": base_event, "clahe_event": clahe_event,
                         "roi_count": roi_count, "fps": fps, "infer_ms": infer_ms,
                         "manual_label": manual_label}, preview)
                    cv2.imshow("AURA | Eye webcam A-B comparison", rendered)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break
                    elif key == ord("o"):
                        manual_label = "Open"
                        print("[LABEL] Open (applies to subsequent samples)")
                    elif key == ord("c"):
                        manual_label = "Closed"
                        print("[LABEL] Closed (applies to subsequent samples)")
                    elif key == ord("u"):
                        manual_label = ""
                        print("[LABEL] Cleared")
                    elif key == ord("r"):
                        baseline_tracker.reset()
                        clahe_tracker.reset()
                        base_duration = clahe_duration = 0.0
                        base_active = clahe_active = False
                    elif key == ord("s"):
                        image_out = args.out / f"eye_ab_capture_{run_id}_{frame_number:06d}.png"
                        cv2.imwrite(str(image_out), rendered)
                        print("[SCREENSHOT]", image_out)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[SAVED]", csv_path)
        print("[NOTE] Manual labels are approximate; score per episode/session, not independent frames.")


if __name__ == "__main__":
    main()
