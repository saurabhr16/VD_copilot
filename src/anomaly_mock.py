"""Temporal / anomaly analysis — mock adapter (PRD §8).

Real signal: normalized frame-difference motion energy + person-count and
person-proximity factors, smoothed with an EMA. Scene-hint scaling makes the
4 demo videos separable; on the A6000 this is replaced by VadCLIPAdapter.
"""
from __future__ import annotations

import cv2
import numpy as np

from .schemas import DetectionItem


class BaseAnomalyScorer:
    name = "base"

    def reset(self, scene_hint: str = "") -> None: ...
    def score_window(self, frames: list[np.ndarray], detections: list[list[DetectionItem]]) -> float:
        raise NotImplementedError


def _scene_gain(scene_hint: str) -> float:
    h = (scene_hint or "").lower()
    if any(k in h for k in ("fight", "assault", "attack")):
        return 1.0
    if any(k in h for k in ("weapon", "knife", "gun", "stab", "shoot")):
        return 0.92
    if any(k in h for k in ("run", "chase", "snatch")):
        return 0.62
    if any(k in h for k in ("normal", "calm", "walk")):
        return 0.42
    return 0.7


class MockAnomalyScorer(BaseAnomalyScorer):
    name = "mock-anom-v1"

    def __init__(self, smoothing_alpha: float = 0.4):
        self.alpha = smoothing_alpha
        self.scene_hint = ""
        self._ema: float | None = None

    def reset(self, scene_hint: str = "") -> None:
        self.scene_hint = scene_hint or ""
        self._ema = None

    def score_window(self, frames: list[np.ndarray], detections: list[list[DetectionItem]]) -> float:
        if len(frames) < 2:
            raw = 0.05
        else:
            diffs = []
            prev = cv2.cvtColor(cv2.resize(frames[0], (160, 90)), cv2.COLOR_BGR2GRAY)
            for f in frames[1:]:
                g = cv2.cvtColor(cv2.resize(f, (160, 90)), cv2.COLOR_BGR2GRAY)
                diffs.append(float(np.abs(g.astype(np.int16) - prev.astype(np.int16)).mean() / 255.0))
                prev = g
            motion = float(np.mean(diffs))  # ~0.003 calm .. ~0.008+ violent (5 fps sampling)
            motion_n = min(1.0, motion / 0.008)

            persons = [sum(1 for d in det if d.label == "person") for det in detections]
            max_persons = max(persons) if persons else 0
            person_f = 0.0 if max_persons == 0 else (0.15 if max_persons == 1 else 0.30)

            prox = 0.0
            for det in detections:
                pb = [d.bbox for d in det if d.label == "person"]
                if len(pb) >= 2:
                    (x1a, y1a, x2a, y2a), (x1b, y1b, x2b, y2b) = pb[0], pb[1]
                    iou = _iou((x1a, y1a, x2a, y2a), (x1b, y1b, x2b, y2b))
                    prox = max(prox, min(0.25, iou * 1.5 + 0.08))

            raw = min(1.0, 0.50 * motion_n + person_f + prox + 0.03)
            # Scene-hint scaling (mock-only stand-in for learned behavior):
            # violent scenes pass through, calm scenes are suppressed.
            gain = _scene_gain(self.scene_hint)
            raw = min(1.0, raw * (0.45 + 0.65 * gain))
            raw = max(raw, min(0.92, motion_n * gain))

        if self._ema is None:
            self._ema = raw
        else:
            self._ema = self.alpha * raw + (1 - self.alpha) * self._ema
        return round(float(self._ema), 3)


def _iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    aa = max(1.0, (a[2] - a[0]) * (a[3] - a[1]))
    bb = max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / (aa + bb - inter)


class VadCLIPAdapter(BaseAnomalyScorer):
    """Placeholder for the VadCLIP/LAVAD temporal model on the A6000.

    Implement score_window() with the real model; the pipeline only depends
    on this interface (PRD §35).
    """

    name = "vadclip-v1"

    def __init__(self, *a, **k):
        raise RuntimeError("VadCLIPAdapter needs a GPU host. See docs/SWAP_TO_REAL_MODELS.md.")

    def reset(self, scene_hint: str = "") -> None: ...
    def score_window(self, frames, detections) -> float: raise NotImplementedError
