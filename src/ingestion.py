"""Video ingestion (PRD §6.1): file-based for the prototype, RTSP-ready.

VideoIngestion probes the source, yields timestamped frames at a target
sampling FPS, and tracks decode health (dropped frames, freezes). The RTSP
path on the deployment host only changes the `source` argument.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class SourceInfo:
    fps: float
    frames: int
    duration_s: float
    width: int
    height: int


def probe(source: str) -> SourceInfo:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video source: {source}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 10.0)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    dur = frames / fps if fps > 0 and frames else 0.0
    return SourceInfo(fps=fps, frames=frames, duration_s=dur, width=w, height=h)


@dataclass
class Frame:
    t: float  # seconds into video
    image: np.ndarray = field(repr=False)


class RollingBuffer:
    """Rolling frame buffer (PRD §12): keeps N pre-event seconds in memory."""

    def __init__(self, max_seconds: float = 8.0):
        self.buf: deque[Frame] = deque()
        self.max_seconds = max_seconds

    def append(self, f: Frame):
        self.buf.append(f)
        while self.buf and (f.t - self.buf[0].t) > self.max_seconds:
            self.buf.popleft()

    def since(self, t: float) -> list[Frame]:
        return [f for f in self.buf if f.t >= t]

    def __len__(self):
        return len(self.buf)


class VideoIngestion:
    def __init__(self, source: str, target_fps: float = 5.0):
        self.source = source
        self.target_fps = target_fps
        self.info = probe(source)
        self.dropped = 0
        self.decoded = 0
        self.frozen_streak = 0

    def iter_frames(self):
        """Yield sampled frames with monotonic timestamps."""
        cap = cv2.VideoCapture(self.source)
        src_fps = self.info.fps or 10.0
        step = max(1, round(src_fps / self.target_fps))
        idx, out_t, prev = 0, 0.0, None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                t = idx / src_fps
                self.decoded += 1
                if prev is not None and np.array_equal(frame[::8, ::8], prev):
                    self.frozen_streak += 1
                else:
                    self.frozen_streak = 0
                    prev = frame[::8, ::8].copy()
                yield Frame(t=t, image=frame)
                out_t = t
            idx += 1
        cap.release()
