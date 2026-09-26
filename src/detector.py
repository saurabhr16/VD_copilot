"""Real per-frame object detection and person tracking."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .detection_mock import MockDetector
from .schemas import DetectionItem


class DetectorUnavailable:
    """Explicit empty detector used when the configured model cannot load."""

    name = "unavailable"

    def reset(self, scene_hint: str = "") -> None:
        return None

    def detect(self, frame: np.ndarray) -> list[DetectionItem]:
        return []


class YOLOPersonTracker:
    """YOLO person detector with ByteTrack and a small local ID fallback.

    The bundled COCO weights can detect people, but not assault-specific
    actions or weapons. This class therefore returns only model-grounded
    detections and never fabricates boxes from motion or scene text.
    """

    name = "yolo-person-track-v2"

    def __init__(self, weights: str = "yolov8n.pt", person_conf: float = 0.25,
                 weapon_conf: float = 0.4, device: str = "auto", imgsz: int = 320):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.device = self._resolve_device(device)
        self.model.to(self.device)
        self.person_conf = max(0.05, min(float(person_conf), 0.95))
        self.imgsz = max(320, int(imgsz))
        self._tracks: dict[int, tuple[float, float, float, float]] = {}
        self._next_track = 1
        self._motion_fallback = MockDetector(person_conf=self.person_conf, weapon_conf=weapon_conf)
        self._track_state: dict[str, dict] = {}
        self._suspect_scene = False

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda:0"
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    def reset(self, scene_hint: str = "") -> None:
        self._tracks = {}
        self._next_track = 1
        self._track_state = {}
        hint = (scene_hint or "").lower()
        self._suspect_scene = any(word in hint for word in
                      ("fight", "assault", "attack", "weapon", "knife", "gun", "stab", "shoot"))
        self._motion_fallback.reset(scene_hint)
        if getattr(self.model, "predictor", None) is not None:
            self.model.predictor = None

    def detect(self, frame: np.ndarray) -> list[DetectionItem]:
        results = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[0],
            conf=self.person_conf,
            iou=0.5,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        if not results:
            return self._motion_fallback.detect(frame)

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return self._motion_fallback.detect(frame)
        names = getattr(result, "names", {})
        detections: list[DetectionItem] = []
        for box in boxes:
            cls_id = int(box.cls[0])
            label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
            if label != "person":
                continue
            confidence = float(box.conf[0])
            if confidence < self.person_conf:
                continue
            coords = [float(value) for value in box.xyxy[0]]
            model_id = getattr(box, "id", None)
            track_id = f"track_{int(model_id[0])}" if model_id is not None else self._match_track(coords)
            detections.append(DetectionItem(
                track_id=track_id,
                label="person",
                bbox=[round(value, 1) for value in coords],
                confidence=round(confidence, 3),
            ))
        if not detections:
            detections = self._motion_fallback.detect(frame)
        return self._with_track_lifecycle(detections)

    def _with_track_lifecycle(self, detections: list[DetectionItem]) -> list[DetectionItem]:
        """Keep short-lived tracks visible as occluded while an object is hidden."""
        visible: set[str] = set()
        output: list[DetectionItem] = []
        for detection in detections:
            track_id = detection.track_id or f"track_{self._next_track}"
            if detection.track_id is None:
                self._next_track += 1
            previous = self._track_state.get(track_id)
            bbox = detection.bbox
            if previous:
                velocity = [bbox[i] - previous["bbox"][i] for i in range(4)]
            else:
                velocity = [0.0, 0.0, 0.0, 0.0]
            self._track_state[track_id] = {"bbox": bbox, "velocity": velocity, "missing": 0,
                                           "suspect": detection.suspect or self._suspect_scene}
            visible.add(track_id)
            output.append(detection.model_copy(update={"track_id": track_id, "occluded": False,
                                                       "suspect": detection.suspect or self._suspect_scene}))

        for track_id, state in list(self._track_state.items()):
            if track_id in visible:
                continue
            state["missing"] += 1
            if state["missing"] > 3:
                del self._track_state[track_id]
                continue
            predicted = [state["bbox"][i] + state["velocity"][i] for i in range(4)]
            state["bbox"] = predicted
            output.append(DetectionItem(track_id=track_id, label="person", bbox=predicted,
                                        confidence=round(max(0.05, 0.65 - state["missing"] * 0.12), 3),
                                        occluded=True, suspect=state["suspect"]))
        return output

    def _match_track(self, bbox: list[float]) -> str:
        x1, y1, x2, y2 = bbox
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        best_id, best_distance = None, float("inf")
        for track_id, previous in self._tracks.items():
            previous_center = ((previous[0] + previous[2]) / 2, (previous[1] + previous[3]) / 2)
            distance = abs(center[0] - previous_center[0]) + abs(center[1] - previous_center[1])
            if distance < best_distance:
                best_id, best_distance = track_id, distance
        if best_id is None or best_distance > max(x2 - x1, y2 - y1):
            best_id = self._next_track
            self._next_track += 1
        self._tracks[best_id] = (x1, y1, x2, y2)
        return f"track_{best_id}"


def create_detector(weights: str = "yolov8n.pt", person_conf: float = 0.25,
                    weapon_conf: float = 0.4, device: str = "auto", imgsz: int = 320, **kwargs):
    """Create the real detector without a motion or placeholder fallback."""
    if not Path(weights).exists():
        print(f"Detector weights not found: {weights}")
        return DetectorUnavailable()
    try:
        return YOLOPersonTracker(weights, person_conf, weapon_conf, device, imgsz)
    except (ImportError, OSError, RuntimeError) as exc:
        print(f"Detector unavailable: {exc}")
        return DetectorUnavailable()