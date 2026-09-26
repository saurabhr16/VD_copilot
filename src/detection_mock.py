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


def create_detector(detector_name: str = "mock-det-v1", **kwargs) -> BaseDetector:
    """Create the configured detector without importing torch in mock mode."""
    if detector_name.startswith("yolov8") or detector_name.startswith("yolo-v8"):
        try:
            return UltralyticsYOLOAdapter(**kwargs)
        except (ImportError, OSError, RuntimeError) as exc:
            print(f"YOLO unavailable, using mock detector: {exc}")
    return MockDetector(person_conf=kwargs.get("person_conf", 0.5),
                        weapon_conf=kwargs.get("weapon_conf", 0.4))


class MockDetector(BaseDetector):
    """Motion-blob person detector + hinted weapon heuristic."""

    name = "mock-det-v1"

    def __init__(self, person_conf: float = 0.5, weapon_conf: float = 0.4, min_area: int = 900):
        self.person_conf = person_conf
        self.weapon_conf = weapon_conf
        self.min_area = min_area
        self.scene_hint = ""
        self._bg = None
        self._bg_frame = None
        self._tracks: dict[int, tuple[float, float]] = {}
        self._next_track = 1

    def reset(self, scene_hint: str = "") -> None:
        self.scene_hint = (scene_hint or "").lower()
        self._bg = cv2.createBackgroundSubtractorMOG2(history=60, varThreshold=25, detectShadows=False)
        self._bg_frame = None
        self._tracks = {}
        self._next_track = 1

    def detect(self, frame: np.ndarray) -> list[DetectionItem]:
        small = cv2.resize(frame, (320, 180))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Seed the background model on the first frame without bailing out. A
        # single real frame can legitimately contain a moving person, and the
        # detected blob should still be emitted instead of returning an empty list.
        if self._bg_frame is None:
            self._bg_frame = gray.copy()
            fg_first = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)[1]
            fg_first = cv2.morphologyEx(fg_first, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            fg_first = cv2.dilate(fg_first, np.ones((5, 5), np.uint8), iterations=1)
            if (fg_first > 0).mean() > 0.001:
                contours, _ = cv2.findContours(fg_first, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    contours = sorted(contours, key=cv2.contourArea, reverse=True)
                    x, y, w, h = cv2.boundingRect(contours[0])
                    if w * h >= self.min_area // 8:
                        x1, y1, x2, y2 = x * (frame.shape[1] / 320.0), y * (frame.shape[0] / 180.0), \
                            (x + w) * (frame.shape[1] / 320.0), (y + h) * (frame.shape[0] / 180.0)
                        return [DetectionItem(track_id=f"track_{self._next_track}", label="person",
                                              bbox=[round(float(x1), 1), round(float(y1), 1),
                                                    round(float(x2), 1), round(float(y2), 1)],
                                              confidence=round(min(0.9, self.person_conf + 0.2), 3))]

        fg_frame = cv2.absdiff(gray, self._bg_frame)
        _, fg = cv2.threshold(fg_frame, 18, 255, cv2.THRESH_BINARY)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        fg = cv2.dilate(fg, np.ones((5, 5), np.uint8), iterations=1)

        # Use MOG2 if it finds motion; otherwise the abs-diff fallback is the real
        # per-frame motion signal and keeps people tracked over time.
        fg_mog = self._bg.apply(gray)
        _, fg_mog = cv2.threshold(fg_mog, 200, 255, cv2.THRESH_BINARY)
        fg_mog = cv2.morphologyEx(fg_mog, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        fg_mog = cv2.dilate(fg_mog, np.ones((5, 5), np.uint8), iterations=1)
        fg = fg_mog if (fg_mog > 0).mean() > 0.002 else fg

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

        # If no motion-driven boxes were found, prefer a real moving-region fallback
        # over the static placeholder pattern.
        if not out and (fg > 0).mean() > 0.002:
            ys, xs = np.where(fg > 0)
            if len(xs) and len(ys):
                x1, x2 = float(xs.min()), float(xs.max())
                y1, y2 = float(ys.min()), float(ys.max())
                x1, y1, x2, y2 = x1 * sx, y1 * sy, x2 * sx, y2 * sy
                out.append(DetectionItem(track_id=f"track_{self._next_track}", label="person",
                                         bbox=[round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                                         confidence=round(min(0.9, self.person_conf + 0.25), 3)))
                self._next_track += 1
                self._tracks[self._next_track - 1] = ((x1 + x2) / 2, (y1 + y2) / 2)

        if self._weapon_hinted() and out and self._motion_level(fg) > 0.012:
            p = out[0]
            x1, y1, x2, y2 = p.bbox
            w = max(14.0, (x2 - x1) * 0.28)
            h = max(10.0, (y2 - y1) * 0.12)
            label = "gun" if "gun" in self.scene_hint or "shoot" in self.scene_hint else "knife"
            out.append(DetectionItem(label=label, bbox=[round(x1, 1), round(y1, 1),
                                                        round(x1 + w, 1), round(y1 + h, 1)],
                                     confidence=round(min(0.85, self.weapon_conf + 0.3), 3)))

        # Keep the background model fresh without drifting too aggressively.
        self._bg_frame = cv2.GaussianBlur(gray, (5, 5), 0)
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

    def __init__(self, weights: str = "yolov8n.pt", person_conf: float = 0.5,
                 weapon_conf: float = 0.4, device: str = "auto"):
        from ultralytics import YOLO  # imported here so mock mode has no torch dep
        import torch

        self.model = YOLO(weights)
        if device == "auto":
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model.to(device)
        self.person_conf = person_conf
        self.weapon_conf = weapon_conf
        self.scene_hint = ""
        self._fallback = MockDetector(person_conf=person_conf, weapon_conf=weapon_conf)

    def reset(self, scene_hint: str = "") -> None:
        self.scene_hint = scene_hint or ""
        self._fallback.reset(scene_hint)
        # Ultralytics stores ByteTrack state on the predictor when persist=True.
        # Clearing it prevents track IDs leaking from one camera into another.
        if getattr(self.model, "predictor", None) is not None:
            self.model.predictor = None

    def detect(self, frame: np.ndarray) -> list[DetectionItem]:
        try:
            res = self.model.track(frame, persist=True, verbose=False,
                                   conf=min(self.person_conf, self.weapon_conf),
                                   tracker="bytetrack.yaml")[0]
        except Exception:
            return self._fallback.detect(frame)

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

        person_labels = {d.label for d in out}
        if not out or not person_labels.intersection({"person", "knife", "gun"}):
            return self._fallback.detect(frame)

        return out
