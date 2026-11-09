"""Fast, localhost-only OpenCV webcam server + latest-frame-only AI worker.

The preview and inference never share a blocking loop. Video travels from the
Windows camera directly to an HTTP MJPEG stream on 127.0.0.1 (not WebRTC/cloud).
No network requests leave the PC. For a remote browser 127.0.0.1 won't work.
"""
from __future__ import annotations

import secrets
import json
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from typing import Optional
from urllib.parse import parse_qs, urlsplit

import cv2
import numpy as np
from PIL import Image


class LatestFrameWorker:
    """Accept snapshots without delaying camera.capture; infer only the newest."""

    def __init__(self, engine, monitor, interval_ms=350, max_width=640):
        self.engine = engine
        self.monitor = monitor
        self.interval = max(0.12, float(interval_ms) / 1000)
        self.max_width = int(max_width)
        self._lock = threading.Lock()
        self._new = threading.Event()
        self._stop = threading.Event()
        self._pending = None
        self._last_offer = 0.0
        self.error = ""
        self.elapsed_ms = 0.0
        self.result_age_ms = 0.0
        self.fps = 0.0
        self._done = deque(maxlen=10)
        self._thread = threading.Thread(target=self._run, name="aura-ai-worker", daemon=True)
        self._thread.start()

    def offer(self, frame_bgr, camera_fps=0.):
        """O(1) for most frames; reserve at most one queued frame."""
        now = time.monotonic()
        if self._stop.is_set() or now - self._last_offer < self.interval:
            return
        self._last_offer = now
        h, w = frame_bgr.shape[:2]
        if self.max_width and w > self.max_width:
            frame_bgr = cv2.resize(frame_bgr, (self.max_width, round(h * self.max_width / w)), interpolation=cv2.INTER_AREA)
        with self._lock:
            self._pending = (frame_bgr.copy(), now, float(camera_fps))
        self._new.set()

    def _run(self):
        while not self._stop.is_set():
            if not self._new.wait(.3):
                continue
            self._new.clear()
            with self._lock:
                item, self._pending = self._pending, None
            if item is None:
                continue
            frame, captured_at, camera_fps = item
            start = time.monotonic()
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(rgb)
                result = self.engine.predict(image)
                now = time.monotonic()
                self.monitor.update(
                    eye=result.get("eye"), yawn=result.get("yawn"), fps=camera_fps,
                    cfg=self.engine.cfg,
                    eye_roi=result.get("eye_processed"),
                    mouth_roi=result.get("mouth"),
                    eye_points=result.get("eye_points"),
                    mouth_points=result.get("mouth_points"),
                )
                self.elapsed_ms = (now - start) * 1000
                self.result_age_ms = (now - captured_at) * 1000
                self._done.append(now)
                self.fps = ((len(self._done) - 1) / (self._done[-1] - self._done[0])
                            if len(self._done) > 1 and self._done[-1] > self._done[0] else 0.)
                self.error = ""
            except Exception as e:
                self.error = f"{type(e).__name__}: {e}"

    def close(self):
        self._stop.set()
        self._new.set()
        self._thread.join(timeout=2.0)


class _ReuseHTTP(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class LocalCameraStream:
    """Windows local webcam -> independent MJPEG browser preview on 127.0.0.1."""

    def __init__(self, camera_index: int, worker: Optional[LatestFrameWorker],
                 width=640, height=480, requested_fps=30, jpeg_quality=65, preview_fps=20):
        self.camera_index = int(camera_index)
        self.worker = worker
        self.width = int(width)
        self.height = int(height)
        self.requested_fps = int(requested_fps)
        self.jpeg_quality = max(40, min(90, int(jpeg_quality)))
        self.preview_fps = max(8, min(30, int(preview_fps)))
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread = None
        self._server = None
        self._server_thread = None
        self._capture = None
        self._latest = None
        self._last_frame = 0.0
        self._last_encoded = 0.0
        self._times = deque(maxlen=40)
        self._last_viewer = time.monotonic()
        self._token = secrets.token_urlsafe(18)
        self.url = ""
        self.events_url = ""
        self.running = False
        self.error = ""

    def _open_capture(self):
        # DirectShow is usually more reliable on Windows for repeated start/stop.
        backends = ([cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
                    if hasattr(cv2, "CAP_DSHOW") else [cv2.CAP_ANY])
        for backend in backends:
            cap = cv2.VideoCapture(self.camera_index, backend)
            if not cap.isOpened():
                cap.release()
                continue
            # Request MJPEG from the device to reduce USB/CPU overhead where supported.
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.requested_fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap
        raise RuntimeError(f"카메라 {self.camera_index}번 열기 실패. 다른 앱의 카메라 점유와 장치 번호를 확인하세요.")

    def start(self):
        with self._lock:
            if self.running:
                return True
            try:
                self._capture = self._open_capture()
                owner = self
                class Handler(BaseHTTPRequestHandler):
                    def log_message(self, _format, *args):
                        return

                    def do_GET(self):
                        req = urlsplit(self.path)
                        if parse_qs(req.query).get("token", [""])[0] != owner._token:
                            self.send_error(404)
                            return
                        if req.path == "/events":
                            monitor = getattr(owner.worker, "monitor", None)
                            events = []
                            if monitor is not None:
                                _, rows, _, _ = monitor.snapshot()
                                events = [{"id": r["id"], "kind": r["kind"]} for r in rows[:16]]
                            payload = json.dumps({"events": events}).encode("utf-8")
                            self.send_response(200)
                            self.send_header("Content-Type", "application/json; charset=utf-8")
                            self.send_header("Cache-Control", "no-store")
                            # Bound to 127.0.0.1, authenticated by a random URL token.
                            # Streamlit HTML-component iframes may use an opaque (null) origin.
                            self.send_header("Access-Control-Allow-Origin", "*")
                            self.send_header("X-Content-Type-Options", "nosniff")
                            self.send_header("Content-Length", str(len(payload)))
                            self.end_headers()
                            self.wfile.write(payload)
                            return
                        if req.path != "/stream":
                            self.send_error(404)
                            return
                        self.send_response(200)
                        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
                        self.send_header("Pragma", "no-cache")
                        self.end_headers()
                        try:
                            while not owner._stop.is_set():
                                with owner._lock:
                                    data = owner._latest
                                if data is not None:
                                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                                    self.wfile.write(b"Content-Length: " + str(len(data)).encode() + b"\r\n\r\n")
                                    self.wfile.write(data + b"\r\n")
                                time.sleep(1. / owner.preview_fps)
                        except (BrokenPipeError, ConnectionResetError, OSError):
                            pass
                self._server = _ReuseHTTP(("127.0.0.1", 0), Handler)
                port = self._server.server_address[1]
                self.url = f"http://127.0.0.1:{port}/stream?token={self._token}"
                self.events_url = f"http://127.0.0.1:{port}/events?token={self._token}"
                self._stop.clear()
                self.running = True
                self.error = ""
                self._server_thread = threading.Thread(target=self._server.serve_forever, name="aura-local-http", daemon=True)
                self._server_thread.start()
                self._thread = threading.Thread(target=self._capture_loop, name="aura-local-cam", daemon=True)
                self._thread.start()
                return True
            except Exception as e:
                self.error = f"{type(e).__name__}: {e}"
                self.running = False
                if self._capture is not None:
                    self._capture.release()
                    self._capture = None
                return False

    def _capture_loop(self):
        bad = 0
        try:
            while not self._stop.is_set():
                cap = self._capture
                if cap is None:
                    break
                ok, frame = cap.read()
                if not ok or frame is None:
                    bad += 1
                    if bad >= 25:
                        self.error = "카메라 프레임을 연속 수신하지 못했어. USB/카메라 점유 여부를 확인해줘."
                        break
                    time.sleep(.04)
                    continue
                bad = 0
                now = time.monotonic()
                self._times.append(now)
                fps = self._camera_fps()
                if self.worker is not None:
                    self.worker.offer(frame, fps)
                # Avoid encoding 30 frames per second when the stream is capped lower.
                if now - self._last_encoded >= 1.0 / self.preview_fps:
                    self._last_encoded = now
                    good, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
                    if good:
                        with self._lock:
                            self._latest = buf.tobytes()
                            self._last_frame = now
        except Exception as e:
            self.error = f"카메라 수신 오류: {type(e).__name__}: {e}"
        finally:
            self.running = False
            if self._capture is not None:
                self._capture.release()
                self._capture = None

    def _camera_fps(self):
        if len(self._times) < 2:
            return 0.0
        span = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / span if span > 0 else 0.

    def snapshot(self):
        with self._lock:
            age = time.monotonic() - self._last_frame if self._last_frame else float("inf")
            return len(self._times), self._camera_fps(), age, self.error

    def heartbeat(self):
        # Public API retained for older 11_streamlit_dashboard code.
        return self.running

    def stop(self):
        self._stop.set()
        server = self._server
        self._server = None
        if server is not None:
            try:
                server.shutdown()
                server.server_close()
            except Exception:
                pass
        if self._server_thread is not None:
            self._server_thread.join(timeout=1.5)
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.running = False
        return self._thread is None or not self._thread.is_alive()
