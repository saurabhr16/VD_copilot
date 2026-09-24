"""Clip recorder (PRD §12): pre/post-event clips linked to incident_id.

Writes with OpenCV (mp4v — always available on CPU), then transcodes to
H264 + faststart via the imageio-ffmpeg binary so clips play in browsers.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import cv2

from .schemas import new_id


def _ffmpeg() -> str | None:
    try:
        import imageio_ffmpeg

        p = imageio_ffmpeg.get_ffmpeg_exe()
        return p if Path(p).exists() else None
    except Exception:
        return None


class ClipRecorder:
    def __init__(self, clips_dir: str | Path, pre_s: float = 5.0, post_s: float = 5.0):
        self.dir = Path(clips_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.pre_s = pre_s
        self.post_s = post_s

    def record(self, source: str, start: float, end: float) -> tuple[str, str, float, float]:
        """Returns (clip_id, clip_path, clip_start, clip_end)."""
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open source for clipping: {source}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 10.0)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        dur = total / fps if fps else end
        cs = max(0.0, start - self.pre_s)
        ce = min(dur, end + self.post_s)
        clip_id = new_id("clip")

        cap.set(cv2.CAP_PROP_POS_FRAMES, int(cs * fps))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        tmp = self.dir / f"{clip_id}.tmp.mp4"
        vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        fidx = int(cs * fps)
        while fidx < int(ce * fps):
            ok, frame = cap.read()
            if not ok:
                break
            vw.write(frame)
            fidx += 1
        vw.release()
        cap.release()

        final = self.dir / f"{clip_id}.mp4"
        if not self._transcode(tmp, final, fps):
            tmp.rename(final)
        else:
            tmp.unlink(missing_ok=True)
        return clip_id, str(final), round(cs, 2), round(ce, 2)

    @staticmethod
    def _transcode(src: Path, dst: Path, fps: float) -> bool:
        exe = _ffmpeg()
        if exe is None or not src.exists():
            return False
        try:
            r = subprocess.run(
                [exe, "-y", "-loglevel", "error", "-i", str(src),
                 "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                 "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(dst)],
                capture_output=True, timeout=120,
            )
            return r.returncode == 0 and dst.exists() and dst.stat().st_size > 0
        except Exception:
            return False
