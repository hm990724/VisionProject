"""Live webcam prototype: MediaPipe FaceMesh + two YOLO26 classification models.

Run:
  python -m pip install opencv-python mediapipe==0.10.21 ultralytics pillow numpy
  python 07_webcam.py
  python 07_webcam.py --camera 1 --yawn-input full

Keys: Q or ESC quit; SPACE pause/resume; D toggle ROI preview.

Important: This is a research/demo prototype, not a driving safety device.
"""
from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
import time

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
from ultralytics import YOLO

from common import to_square
from roi_utils import crop_eyes

ROOT = Path(__file__).resolve().parent
DEFAULT_EYES = ROOT / "runs/eyes_yolo26n-cls/weights/best.pt"
DEFAULT_YAWN = ROOT / "runs/yawn_yolo26n-cls/weights/best.pt"
# Outer facial outline indices in MediaPipe FaceMesh. Bounding all landmarks yields a face ROI.
FACE_OUTLINE = (10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109)


def face_crop(frame_rgb: np.ndarray, landmarks, pad: float = 0.15):
    """Crop face from RGB frame, preserving some surroundings. Returns PIL or None."""
    h, w = frame_rgb.shape[:2]
    xs = [landmarks[i].x * w for i in FACE_OUTLINE]
    ys = [landmarks[i].y * h for i in FACE_OUTLINE]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    dx, dy = (x1 - x0) * pad, (y1 - y0) * pad
    x0, y0 = max(0, int(x0 - dx)), max(0, int(y0 - dy))
    x1, y1 = min(w, int(x1 + dx)), min(h, int(y1 + dy))
    if x1 - x0 < 20 or y1 - y0 < 20:
        return None
    return Image.fromarray(frame_rgb[y0:y1, x0:x1].copy())


def label_id(model, label):
    values = {str(v).lower(): k for k, v in model.names.items()}
    if label.lower() not in values:
        raise ValueError(f"Model expected class '{label}', found: {model.names}")
    return values[label.lower()]


def predict_probability(model, image: Image.Image, class_index: int) -> float:
    pred = model.predict(to_square(image, 224), imgsz=224, verbose=False)[0]
    return float(pred.probs.data[class_index].item())


def render_preview(canvas, image, x, y, label, w=135, h=95):
    """Small crop preview; grayscale placeholders avoided."""
    if image is None:
        return
    thumb = image.convert("RGB")
    thumb.thumbnail((w, h))
    arr = cv2.cvtColor(np.asarray(thumb), cv2.COLOR_RGB2BGR)
    hh, ww = arr.shape[:2]
    if x + ww <= canvas.shape[1] and y + hh <= canvas.shape[0]:
        canvas[y:y + hh, x:x + ww] = arr
        cv2.putText(canvas, label, (x, min(canvas.shape[0] - 8, y + hh + 17)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description="Webcam drowsiness signal demo")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--eyes-weights", type=Path, default=DEFAULT_EYES)
    ap.add_argument("--yawn-weights", type=Path, default=DEFAULT_YAWN)
    ap.add_argument("--yawn-input", choices=("face", "full", "mouth"), default="face",
                    help="Use face (default), whole frame, or mouth crop. Compare against training samples.")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--eye-threshold", type=float, default=0.5)
    ap.add_argument("--yawn-threshold", type=float, default=0.65)
    ap.add_argument("--closed-seconds", type=float, default=2.0)
    ap.add_argument("--yawn-seconds", type=float, default=0.8)
    ap.add_argument("--yawn-cooldown", type=float, default=4.0)
    ap.add_argument("--stride", type=int, default=2, help="Classify every Nth frame; landmark detection every frame")
    args = ap.parse_args()
    if args.stride < 1:
        ap.error("--stride must be >= 1")
    for weights in (args.eyes_weights, args.yawn_weights):
        if not weights.is_file():
            ap.error(f"Missing model weights: {weights}")

    eyes_model = YOLO(str(args.eyes_weights))
    yawn_model = YOLO(str(args.yawn_weights))
    closed_id = label_id(eyes_model, "Closed")
    yawn_id = label_id(yawn_model, "yawn")
    print("Eye classes:", eyes_model.names, "| Yawn classes:", yawn_model.names)
    print("[Q/ESC] Quit  [SPACE] Pause  [D] Toggle ROI previews")
    print("Yawn input:", args.yawn_input, "(validate this crop choice with actual training samples)")

    # CAP_DSHOW is particularly useful for Windows USB webcams; fallback to auto.
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW if __import__('sys').platform == 'win32' else cv2.CAP_ANY)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Cannot open webcam. Try --camera 1 or check camera privacy settings.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    last_eye = None
    last_yawn = None
    closed_since = None
    yawn_since = None
    last_yawn_event = float('-inf')
    yawn_events = 0
    yawn_latched = False
    fps_window = deque(maxlen=30)
    frame_index = 0
    paused = False
    show_rois = True
    last_crops = []
    last_yawn_crop = None
    last_valid = None

    try:
        with mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        ) as mesh:
            while True:
                tic = time.perf_counter()
                if not paused:
                    ok, frame = cap.read()
                    if not ok:
                        print("Cannot read camera frame")
                        break
                    frame_index += 1
                    frame = cv2.flip(frame, 1)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil = Image.fromarray(rgb)
                    detection = mesh.process(rgb)
                    found = bool(detection.multi_face_landmarks)
                    if found:
                        lm = detection.multi_face_landmarks[0].landmark
                        pts = [(int(p.x * frame.shape[1]), int(p.y * frame.shape[0])) for p in lm]
                        for idx in (33, 133, 362, 263, 61, 291):
                            cv2.circle(frame, pts[idx], 2, (90, 210, 255), -1)
                    if found and frame_index % args.stride == 0:
                        eyes = crop_eyes(pil, mesh)  # current shared ROI function
                        last_crops = eyes
                        if eyes:
                            last_eye = sum(predict_probability(eyes_model, eye, closed_id) for eye in eyes) / len(eyes)
                        else:
                            last_eye = None
                        if args.yawn_input == 'full':
                            ycrop = pil
                        elif args.yawn_input == 'face':
                            ycrop = face_crop(rgb, lm)
                        else:
                            # Optional mouth ROI: only useful if yawn model was trained on mouth close-ups.
                            xx = [lm[i].x * rgb.shape[1] for i in (61, 291, 0, 17, 13, 14)]
                            yy = [lm[i].y * rgb.shape[0] for i in (61, 291, 0, 17, 13, 14)]
                            mx = (min(xx) + max(xx)) / 2
                            my = (min(yy) + max(yy)) / 2
                            half = max(max(xx) - min(xx), max(yy) - min(yy)) * 0.85
                            x0, y0 = max(0, int(mx-half)), max(0, int(my-half))
                            x1, y1 = min(rgb.shape[1], int(mx+half)), min(rgb.shape[0], int(my+half))
                            ycrop = Image.fromarray(rgb[y0:y1, x0:x1].copy()) if x1 > x0 and y1 > y0 else None
                        last_yawn_crop = ycrop
                        last_yawn = predict_probability(yawn_model, ycrop, yawn_id) if ycrop else None
                        last_valid = time.monotonic()
                    elif not found:
                        # Detection failure is UNKNOWN, not Open/no_yawn. Do not count it as safe.
                        last_eye = last_yawn = None
                        closed_since = yawn_since = None
                        yawn_latched = False
                        last_crops = []
                        last_yawn_crop = None
                    elif last_valid is not None and time.monotonic() - last_valid > 0.8:
                        last_eye = last_yawn = None
                        closed_since = yawn_since = None
                        yawn_latched = False

                    now = time.monotonic()
                    if last_eye is not None and last_eye >= args.eye_threshold:
                        if closed_since is None:
                            closed_since = now
                    else:
                        closed_since = None
                    closed_duration = now - closed_since if closed_since is not None else 0.0
                    if last_yawn is not None and last_yawn >= args.yawn_threshold:
                        if yawn_since is None:
                            yawn_since = now
                        if (not yawn_latched and now - yawn_since >= args.yawn_seconds
                                and now - last_yawn_event >= args.yawn_cooldown):
                            yawn_events += 1
                            last_yawn_event = now
                            yawn_latched = True
                    else:
                        yawn_since = None
                        yawn_latched = False
                    alert = closed_duration >= args.closed_seconds
                    display = frame.copy()
                    fps_window.append(time.perf_counter() - tic)
                else:
                    display = paused_frame.copy()
                    alert = False
                    closed_duration = 0.0
                    found = None

                def put(text, row, color=(255, 255, 255)):
                    cv2.putText(display, text, (12, row), cv2.FONT_HERSHEY_SIMPLEX,
                                0.65, (0, 0, 0), 4, cv2.LINE_AA)
                    cv2.putText(display, text, (12, row), cv2.FONT_HERSHEY_SIMPLEX,
                                0.65, color, 2, cv2.LINE_AA)

                put("FACE: " + ("DETECTED" if found else "UNKNOWN" if found is False else "PAUSED"), 28)
                eye_text = f"EYES: {'CLOSED' if last_eye >= args.eye_threshold else 'OPEN'} ({last_eye:.0%} closed)" if last_eye is not None else "EYES: UNKNOWN"
                yawn_text = f"YAWN: {'YES' if last_yawn >= args.yawn_threshold else 'NO'} ({last_yawn:.0%})" if last_yawn is not None else "YAWN: UNKNOWN"
                put(eye_text, 58, (0, 255, 255) if last_eye is not None and last_eye >= args.eye_threshold else (255, 255, 255))
                put(yawn_text, 88)
                put(f"Closed: {closed_duration:.1f}s | Yawns: {yawn_events}", 118)
                if alert:
                    cv2.rectangle(display, (2, 2), (display.shape[1] - 3, display.shape[0] - 3), (0, 0, 255), 7)
                    put("DROWSINESS WARNING", 150, (0, 0, 255))
                if fps_window:
                    put(f"FPS ~ {len(fps_window) / max(sum(fps_window), 1e-6):.1f}", display.shape[0] - 14)
                if show_rois and display.shape[1] >= 360 and display.shape[0] >= 360:
                    preview_y = 175
                    for i, eye in enumerate(last_crops[:2]):
                        render_preview(display, eye, 12 + i * 145, preview_y, f"Eye {i+1}")
                    render_preview(display, last_yawn_crop, 12, preview_y + 125, "Yawn input")
                cv2.imshow("VisionProject - webcam demo (Q quit / SPACE pause / D crops)", display)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    break
                if key == ord('d'):
                    show_rois = not show_rois
                if key == 32:
                    paused = not paused
                    if paused:
                        paused_frame = display.copy()
                    else:
                        # Do not infer continued closure across a pause.
                        closed_since = yawn_since = None
                        yawn_latched = False
                        last_eye = last_yawn = None
                        last_valid = None
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
