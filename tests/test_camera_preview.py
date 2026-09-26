import sys
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.main import app, db
from src.schemas import DetectionItem
from src.store import CameraRow


def test_dashboard_route_serves_utf8_html():
    client = TestClient(app)
    resp = client.get("/")

    assert resp.status_code == 200
    assert "Violence &amp; Anomaly Detection" in resp.text
    assert "🎥" in resp.text


def test_camera_analysis_aggregates_objects_across_frames(monkeypatch, tmp_path):
    class FakeFrame:
        def __init__(self, image):
            self.image = image

    class FakeDetector:
        def reset(self, scene_hint):
            self.scene_hint = scene_hint

        def detect(self, frame):
            if frame == "frame_1":
                return [
                    DetectionItem(track_id="t1", label="person", bbox=[1, 2, 3, 4], confidence=0.81),
                    DetectionItem(track_id="t2", label="person", bbox=[10, 11, 12, 13], confidence=0.76),
                ]
            return [
                DetectionItem(track_id="t3", label="knife", bbox=[20, 21, 22, 23], confidence=0.84),
            ]

    class FakeIngestion:
        def __init__(self, source, target_fps):
            self.source = source
            self.target_fps = target_fps

        def iter_frames(self):
            yield FakeFrame("frame_1")
            yield FakeFrame("frame_2")

    source = tmp_path / "cam_multi.mp4"
    source.write_bytes(b"fake-video-bytes")

    with db.session() as s:
        row = s.get(CameraRow, "cam_multi_test")
        if row is None:
            s.add(CameraRow(
                camera_id="cam_multi_test",
                source_path=str(source),
                fps=10,
                frames=2,
                duration_s=0.2,
                width=640,
                height=360,
                status="uploaded",
                extra={"scene_hint": "fight assault weapon"},
            ))
        else:
            row.source_path = str(source)
            row.extra = {"scene_hint": "fight assault weapon"}
        s.commit()

    monkeypatch.setattr("api.main.create_detector", lambda *args, **kwargs: FakeDetector())
    monkeypatch.setattr("api.main.VideoIngestion", FakeIngestion)

    client = TestClient(app)
    resp = client.get("/api/cameras/cam_multi_test/analysis")

    assert resp.status_code == 200
    labels = [obj["label"] for obj in resp.json()["detections"]]
    assert "person" in labels
    assert "knife" in labels
    assert len(resp.json()["detections"]) >= 3


def test_yolo_detector_falls_back_when_real_model_detects_nothing(monkeypatch):
    class FakeYOLOModel:
        def to(self, device):
            return self

    class FakeTrackResult:
        boxes = []

    class FakeYOLO:
        def __init__(self, *args, **kwargs):
            self.predictor = None
            self.track_calls = 0

        def to(self, device):
            return self

        def track(self, *args, **kwargs):
            self.track_calls += 1
            return [FakeTrackResult()]

    monkeypatch.setattr("src.detection_mock.YOLO", FakeYOLO, raising=False)

    det = __import__("src.detection_mock", fromlist=["UltralyticsYOLOAdapter"]).UltralyticsYOLOAdapter(
        weights="yolov8n.pt", person_conf=0.5, weapon_conf=0.4, device="cpu"
    )
    det.reset("fight assault weapon")
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[30:80, 15:85] = 255
    out = det.detect(frame)
    assert any(d.label == "person" for d in out)
    assert any(d.track_id for d in out if d.label == "person")


def test_camera_preview_route_serves_source_video(tmp_path):
    source = tmp_path / "cam_preview.mp4"
    source.write_bytes(b"fake-video-bytes")

    with db.session() as s:
        row = s.get(CameraRow, "cam_preview_test")
        if row is None:
            s.add(
                CameraRow(
                    camera_id="cam_preview_test",
                    source_path=str(source),
                    fps=30,
                    frames=1,
                    duration_s=1.0,
                    width=640,
                    height=360,
                    status="uploaded",
                    extra={"scene_hint": "preview"},
                )
            )
        else:
            row.source_path = str(source)
        s.commit()

    client = TestClient(app)
    resp = client.get("/api/cameras/cam_preview_test/video")

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("video/")


def test_camera_analysis_includes_boxes_and_story(tmp_path):
    source = tmp_path / "cam_story.mp4"
    source.write_bytes(b"fake-video-bytes")

    with db.session() as s:
        row = s.get(CameraRow, "cam_story_test")
        if row is None:
            s.add(
                CameraRow(
                    camera_id="cam_story_test",
                    source_path=str(source),
                    fps=10,
                    frames=10,
                    duration_s=1.0,
                    width=640,
                    height=360,
                    status="uploaded",
                    extra={"scene_hint": "fight assault"},
                )
            )
        else:
            row.source_path = str(source)
            row.extra = {"scene_hint": "fight assault"}
        s.commit()

    client = TestClient(app)
    resp = client.get("/api/cameras/cam_story_test/analysis")

    assert resp.status_code == 200
    body = resp.json()
    assert body["camera_id"] == "cam_story_test"
    assert body["story"]
    assert body["detections"] == []
    assert body["frames"] == []
