"""AURA DRIVE safety-response *simulation only*.

No vehicle hardware, geolocation, emergency dispatch, or external requests.
Only an explicit, button-confirmed simulated response can end a check.
"""
from __future__ import annotations
from collections import deque
import threading
import time
from datetime import datetime


class SafetySimulation:
    """Thread-safe, deterministic monotonic-time simulation.

    WARNING: the values are demo rules, NOT validated driving intervention rules.
    """

    def __init__(self):
        self.lock = threading.RLock()
        self.eye_events = deque()
        self.activity_log = deque(maxlen=100)
        self.mode = "monitor"
        self.speed = 80.0  # entirely virtual km/h
        self.window_open = False
        self.stop_start = None
        self.rest_start = None
        self.unresponsive_start = None
        self.demo_fast = False
        self.acknowledged = False
        self.last_eye_event = None
        self._last_tick = time.monotonic()
        self._log("시뮬레이션 준비", "실제 차량 제어/신고 없음")

    def _log(self, name, detail=""):
        self.activity_log.appendleft({"시간": datetime.now().strftime("%H:%M:%S"), "상태": name, "내용": detail})

    def configure(self, demo_fast=False):
        with self.lock:
            self.demo_fast = bool(demo_fast)

    def on_eye_warning(self, now=None):
        now = time.monotonic() if now is None else now
        with self.lock:
            if self.mode != "monitor":
                return
            self.eye_events.append(now)
            while self.eye_events and now - self.eye_events[0] > 300:
                self.eye_events.popleft()
            self.last_eye_event = now
            self._log("눈 감김 경고", f"최근 5분 {len(self.eye_events)}회")
            if len(self.eye_events) >= 2:
                self._begin_stop(now)

    def _begin_stop(self, now):
        self.mode = "stopping"
        self.speed = 80.0
        self.window_open = False
        self.stop_start = now
        self.rest_start = None
        self.unresponsive_start = None
        self.acknowledged = False
        self._last_tick = now
        self._log("가상 안전 정차", "반복 감지 2회: 화면 속 속도만 감속")

    def start_stop_demo(self, now=None):
        now = time.monotonic() if now is None else now
        with self.lock:
            self.eye_events.clear()
            self._begin_stop(now)

    def start_unresponsive_demo(self, now=None):
        now = time.monotonic() if now is None else now
        with self.lock:
            self.mode = "unresponsive"
            self.unresponsive_start = now
            self.stop_start = None
            self.rest_start = None
            self.window_open = False
            self.speed = 0.0
            self.acknowledged = False
            self._last_tick = now
            self._log("무반응 시나리오", "실제 움직임/의식 상태를 측정하지 않는 수동 시뮬레이션")

    def respond(self, now=None):
        now = time.monotonic() if now is None else now
        with self.lock:
            if self.mode in {"rest", "unresponsive", "sos", "stopping"}:
                self.mode = "ready"
                self.speed = 0.0
                self.window_open = False
                self.unresponsive_start = None
                self.acknowledged = True
                self.eye_events.clear()
                self._log("운전자 응답", "가상 정차 유지 · 재출발은 별도 승인")

    def reset(self, now=None):
        now = time.monotonic() if now is None else now
        with self.lock:
            self.mode = "monitor"
            self.speed = 80.0
            self.window_open = False
            self.stop_start = None
            self.rest_start = None
            self.unresponsive_start = None
            self.eye_events.clear()
            self.acknowledged = False
            self._last_tick = now
            self._log("시뮬레이션 초기화", "모니터링 복귀")

    def tick(self, now=None):
        now = time.monotonic() if now is None else now
        with self.lock:
            dt = max(0.0, min(2.0, now - self._last_tick))
            self._last_tick = now
            if self.mode == "stopping":
                self.speed = max(0.0, self.speed - 16.0 * dt)
                if self.speed <= 0:
                    self.mode = "rest"
                    self.rest_start = now
                    self.window_open = True  # virtual action only
                    self._log("가상 정차 완료", "가상 창문 개방 · 휴식 안내 · 실제 차량과 무관")
            elif self.mode == "rest":
                required = 15.0 if self.demo_fast else 300.0
                if self.rest_start is not None and now - self.rest_start >= required:
                    self.mode = "ready"
                    self.window_open = False
                    self._log("휴식 시간 종료", "자동 재출발 없음 · 운전자 확인 필요")
            elif self.mode == "unresponsive":
                required = 30.0 if self.demo_fast else 600.0
                if self.unresponsive_start is not None and now - self.unresponsive_start >= required:
                    self.mode = "sos"
                    self._log("모의 119 알림", "실제 신고/통화 전혀 없음")
            return self.snapshot(now)

    def snapshot(self, now=None):
        now = time.monotonic() if now is None else now
        with self.lock:
            while self.eye_events and now - self.eye_events[0] > 300:
                self.eye_events.popleft()
            rest_duration = 15.0 if self.demo_fast else 300.0
            response_duration = 30.0 if self.demo_fast else 600.0
            rest_remaining = max(0., rest_duration - (now - self.rest_start)) if self.rest_start is not None and self.mode == "rest" else 0.
            response_remaining = max(0., response_duration - (now - self.unresponsive_start)) if self.unresponsive_start is not None and self.mode == "unresponsive" else 0.
            return {"mode": self.mode, "speed": self.speed, "window_open": self.window_open,
                    "recent_eye_events": len(self.eye_events), "rest_remaining": rest_remaining,
                    "response_remaining": response_remaining, "demo_fast": self.demo_fast,
                    "events": list(self.activity_log), "acknowledged": self.acknowledged}
