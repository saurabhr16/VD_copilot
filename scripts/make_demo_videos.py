"""Generate 4 synthetic surveillance demo videos (no external assets).

  cam01_normal  — person walking slowly (low motion, expect: no incident)
  cam02_fight   — two people converging + rapid oscillation (expect: fighting/confirmed)
  cam03_weapon  — person + knife-like object with glint (expect: stabbing/confirmed)
  cam04_running — person sprinting across frame (expect: uncertain/low or none)

Usage:  python scripts/make_demo_videos.py [--seconds 20]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

W, H, FPS = 640, 360, 10


def background(scene: str) -> np.ndarray:
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[:] = (38, 40, 44)
    cv2.rectangle(img, (0, H // 2), (W, H), (52, 54, 58), -1)  # floor
    for x in range(0, W, 80):  # floor perspective lines
        cv2.line(img, (W // 2, H // 2), (x, H), (60, 62, 66), 1)
    cv2.rectangle(img, (0, 0), (W, H // 2), (30, 32, 40), -1)  # wall
    if scene == "night":
        img = (img * 0.55).astype(np.uint8)
    return img


def person(img: np.ndarray, cx: float, cy: float, scale: float, color: tuple,
           lean: float = 0.0):
    w, h = int(34 * scale), int(88 * scale)
    x, y = int(cx - w / 2 + lean), int(cy - h / 2)
    cv2.rectangle(img, (x, y + int(22 * scale)), (x + w, y + h), color, -1)
    cv2.circle(img, (x + w // 2, y + int(14 * scale)), int(12 * scale), color, -1)
    # legs hint
    cv2.rectangle(img, (x + 3, y + h - int(20 * scale)),
                  (x + w // 2 - 1, y + h), (color[0] // 2, color[1] // 2, color[2] // 2), -1)
    cv2.rectangle(img, (x + w // 2 + 1, y + h - int(20 * scale)),
                  (x + w - 3, y + h), (color[0] // 2, color[1] // 2, color[2] // 2), -1)


def knife(img: np.ndarray, x: float, y: float, glint: float):
    pts = np.array([[x, y], [x + 34, y - 8], [x + 38, y - 3], [x + 4, y + 6]], np.int32)
    blade = int(150 + 90 * glint)
    cv2.fillPoly(img, [pts], (blade, blade, min(255, blade + 20)))
    cv2.rectangle(img, (int(x) - 8, int(y) - 2), (int(x) + 2, int(y) + 6), (20, 20, 20), -1)


def overlay(img: np.ndarray, cam: str, t: float):
    cv2.putText(img, cam, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)
    cv2.putText(img, f"T+{t:05.1f}s", (12, H - 14), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (200, 200, 200), 1)


def write(path: Path, draw):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(path), fourcc, FPS, (W, H))
    if not vw.isOpened():
        raise RuntimeError("VideoWriter failed to open")
    n = int(SECONDS * FPS)
    for i in range(n):
        img = draw(i / FPS, n / FPS)
        vw.write(img)
    vw.release()


def draw_normal(t: float, dur: float) -> np.ndarray:
    img = background("day")
    cx = 80 + (W - 160) * (t / dur)  # slow stroll left -> right
    bob = 2 * np.sin(2 * np.pi * t * 0.8)
    person(img, cx, 235 + bob, 1.0, (90, 140, 200))
    overlay(img, "cam01 - lobby (normal)", t)
    return img


def draw_fight(t: float, dur: float) -> np.ndarray:
    img = background("day")
    # Phase 1 (0-30%): approach; Phase 2: rapid oscillation + overlap; Phase 3: separate slightly
    if t < dur * 0.3:
        k = t / (dur * 0.3)
        ax, bx = 140 + 140 * k, 500 - 140 * k
        j = 0.0
    elif t < dur * 0.8:
        k = (t - dur * 0.3) / (dur * 0.5)
        j = 26 * np.sin(2 * np.pi * t * 3.2) * (0.6 + 0.4 * k)
        ax, bx = 280 + j, 360 - j
    else:
        ax, bx = 250, 390
        j = 8 * np.sin(2 * np.pi * t * 2.0)
        ax, bx = ax + j, bx - j
    person(img, ax, 232 + 3 * np.sin(2 * np.pi * t * 3.2), 1.05, (60, 60, 220))
    person(img, bx, 238 - 3 * np.sin(2 * np.pi * t * 3.2), 1.05, (220, 90, 60))
    overlay(img, "cam02 - parking (fight)", t)
    return img


def draw_weapon(t: float, dur: float) -> np.ndarray:
    img = background("night")
    cx = W // 2 + 60 * np.sin(2 * np.pi * t / dur)
    person(img, cx, 235, 1.1, (120, 200, 120))
    glint = max(0.0, np.sin(2 * np.pi * t * 1.4)) ** 2
    knife(img, cx + 22, 218 + 4 * np.sin(2 * np.pi * t * 2.0), glint)
    if t > dur * 0.35:  # second person backs away
        person(img, cx + 150, 240, 0.95, (200, 170, 90))
    overlay(img, "cam03 - alley (weapon)", t)
    return img


def draw_running(t: float, dur: float) -> np.ndarray:
    img = background("day")
    laps = 3
    k = (t / dur * laps) % 1.0
    cx = -40 + (W + 80) * k  # fast crossing, repeated
    person(img, cx, 235, 1.0, (200, 120, 60), lean=10)
    overlay(img, "cam04 - corridor (running)", t)
    return img


SECONDS = float(os.environ.get("VSD_DEMO_SECONDS", "20"))

SCENES = [
    ("cam01", "cam01_normal.mp4", "normal walk calm", draw_normal),
    ("cam02", "cam02_fight.mp4", "fight assault", draw_fight),
    ("cam03", "cam03_weapon.mp4", "weapon knife stab", draw_weapon),
    ("cam04", "cam04_running.mp4", "run chase", draw_running),
]


def ensure_demo_videos(out_dir: str | Path) -> list[tuple[str, Path, str]]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(11)  # noqa: F841 (reserved for noise overlays)
    result = []
    for cam_id, fname, hint, draw in SCENES:
        p = out_dir / fname
        if not p.exists() or p.stat().st_size < 10000:
            write(p, draw)
        result.append((cam_id, p, hint))
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=SECONDS)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "demo_videos"))
    args = ap.parse_args()
    SECONDS = args.seconds
    for cam_id, p, hint in ensure_demo_videos(args.out):
        print(f"{cam_id}: {p} ({p.stat().st_size / 1024:.0f} KB) hint=[{hint}]")
    sys.exit(0)
