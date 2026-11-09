"""Testable, model-independent continuous detection logic for AURA DRIVE.

The model score is a classification score, not a clinical risk probability.
An event is emitted ONCE when consecutive above-threshold observations span
at least the user-configured seconds. Below-threshold/unknown resets timing.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class Condition:
    since: float | None = None
    latched: bool = False
    duration: float = 0.0

    def update(self, score: float | None, threshold: float, seconds: float, now: float) -> bool:
        if score is None or score < threshold:
            self.since = None
            self.latched = False
            self.duration = 0.0
            return False
        if self.since is None:
            self.since = now
        self.duration = max(0.0, now - self.since)
        if self.duration >= seconds and not self.latched:
            self.latched = True
            return True
        return False


class ContinuousAlerts:
    def __init__(self, max_gap: float = 1.6):
        self.eye = Condition()
        self.yawn = Condition()
        self.previous_at: float | None = None
        self.max_gap = max_gap

    def reset(self):
        self.eye = Condition()
        self.yawn = Condition()
        self.previous_at = None

    def update(self, eye_score: float | None, yawn_score: float | None,
               cfg: dict, now: float) -> list[str]:
        if self.previous_at is not None and now - self.previous_at > self.max_gap:
            self.eye = Condition()
            self.yawn = Condition()
        self.previous_at = now
        eye_triggered = self.eye.update(eye_score, cfg['eye'], cfg['eye_seconds'], now)
        yawn_triggered = self.yawn.update(yawn_score, cfg['yawn'], cfg['yawn_seconds'], now)
        # Eye is the primary alert if both complete in the same frame.
        return (['eye'] if eye_triggered else []) + (['yawn'] if yawn_triggered else [])
