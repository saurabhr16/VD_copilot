"""Object detection — mock/heuristic adapter (PRD §6.2).

Real signal: motion blobs (background subtraction) -> person boxes with a
tiny centroid tracker. Weapon boxes are a clearly-marked heuristic: when the
scene label suggests a weapon video AND fast motion overlaps a person box, a
knife/gun box is emitted. On the A6000 this module is replaced by
UltralyticsYOLOAdapter (same input/output contract, PRD §35).
"""
from __future__ import annotations

import cv2
import numpy as np

from .schemas import DetectionItem


class BaseDetector:
    name = "base"

    def reset(self, scene_hint: str = "") -> None: ...
    def detect(self, frame: np.ndarray) -> list[DetectionItem]: raise NotImplementedError


class MockDetector(BaseDetector):
    """Motion-blob person detector + hinted weapon heuristic."""

    name = "mock-det-v1"

    def __init__(self, person_conf: float = 0.5, weapon_conf: float = 0.4, min_area: int = 900):
        self.person_conf = person_conf
        self.weapon_conf = weapon_conf
        self.min_area = min_area
        self.scene_hint = ""
        self._bg = None
        self._tracks: dict[int, tuple[float, float]] = {}
        self._next_track = 1

    def reset(self, scene_hint: str = "") -> None:
        self.scene_hint = (scene_hint or "").lower()
        self._bg = cv2.createBackgroundSubtractorMOG2(history=60, varThreshold=25, detectShadows=False)
        self._tracks = {}
        self._next_track = 1

    def detect(self, frame: np.ndarray) -> list[DetectionItem]:
        small = cv2.resize(frame, (320, 180))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        fg = self._bg.apply(gray)
        _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        fg = cv2.dilate(fg, np.ones((5, 5), np.uint8), iterations=1)
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        H, W = frame.shape[:2]
        sx, sy = W / 320.0, H / 180.0
        boxes = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w * h < self.min_area // 4:
                continue
            if h < 12 or w < 12:
                continue
            x1, y1, x2, y2 = x * sx, y * sy, (x + w) * sx, (y + h) * sy
            if x1 <= 0 and y1 <= 0 and x2 >= W * 0.9 and y2 >= H * 0.9:
                continue
            boxes.append([x1, y1, x2, y2, w * h])
        boxes.sort(key=lambda b: -b[4])
        boxes = boxes[:6]

        out: list[DetectionItem] = []
        used_tracks = set()
        for x1, y1, x2, y2, area in boxes:
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            best, best_d = None, 90.0
            for tid, (px, py) in self._tracks.items():
                if tid in used_tracks:
                    continue
                d = abs(cx - px) + abs(cy - py)
                if d < best_d:
                    best, best_d = tid, d
            if best is None:
                best = self._next_track
                self._next_track += 1
            self._tracks[best] = (cx, cy)
            used_tracks.add(best)
            conf = min(0.93, self.person_conf + min(0.4, area / 40000.0))
            out.append(DetectionItem(track_id=f"track_{best}", label="person",
                                     bbox=[round(float(x1), 1), round(float(y1), 1),
                                           round(float(x2), 1), round(float(y2), 1)],
                                     confidence=round(float(conf), 3)))
        # Weapon heuristic (mock only — real system uses a trained detector).
        if self._weapon_hinted() and out and self._motion_level(fg) > 0.012:
            p = out[0]
            x1, y1, x2, y2 = p.bbox
            w = max(14.0, (x2 - x1) * 0.28)
            h = max(10.0, (y2 - y1) * 0.12)
            label = "gun" if "gun" in self.scene_hint or "shoot" in self.scene_hint else "knife"
            out.append(DetectionItem(label=label, bbox=[round(x1, 1), round(y1, 1),
                                                        round(x1 + w, 1), round(y1 + h, 1)],
                                     confidence=round(min(0.85, self.weapon_conf + 0.3), 3)))
        return out

    # -- helpers -----------------------------------------------------------
    def _weapon_hinted(self) -> bool:
        return any(k in self.scene_hint for k in ("weapon", "knife", "gun", "stab", "shoot"))

    @staticmethod
    def _motion_level(fg: np.ndarray) -> float:
        return float((fg > 0).mean())


class UltralyticsYOLOAdapter(BaseDetector):
    """Drop-in YOLO adapter for the A6000 host (PRD §35).

    Install: pip install ultralytics torch --index-url <cu121>.
    Keeps the exact same detect() contract as MockDetector.
    """

    name = "yolo-v1"

    def __init__(self, weights: str = "yolov8m.pt", person_conf: float = 0.5,
                 weapon_conf: float = 0.4, device: str = "cuda:0"):
        from ultralytics import YOLO  # imported here so mock mode has no torch dep

        self.model = YOLO(weights)
        self.model.to(device)
        self.person_conf = person_conf
        self.weapon_conf = weapon_conf
        self.scene_hint = ""

    def reset(self, scene_hint: str = "") -> None:
        self.scene_hint = scene_hint or ""

    def detect(self, frame: np.ndarray) -> list[DetectionItem]:
        res = self.model.predict(frame, verbose=False, conf=min(self.person_conf, self.weapon_conf))[0]
        out = []
        for b in res.boxes:
            cls = res.names[int(b.cls[0])]
            conf = float(b.conf[0])
            thr = self.weapon_conf if cls in ("knife", "gun") else self.person_conf
            if conf < thr:
                continue
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
            tid = f"track_{int(b.id[0])}" if b.id is not None else None
            out.append(DetectionItem(track_id=tid, label=cls,
                                     bbox=[x1, y1, x2, y2], confidence=round(conf, 3)))
        return out
